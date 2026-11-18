//! Spend caps for any [`LLM`]: refuse a call before it is sent when it would not fit.
//!
//! [`BudgetedLLM`] asks a [`SpendGate`] to reserve the most a call can cost (the prompt
//! tokens plus `max_tokens`, or a default output reserve) before the call is sent, and
//! hands the usage the provider reports back to the gate when the call returns. The gate
//! decides what a token costs and where the limit is, so the same wrapper works for a
//! token count, a USD cap, or a budget shared by chains and agents running in parallel.
//!
//! A gate is two small traits, so any budget or pricing crate can plug in:
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use langchain_rust::llm::{BudgetedLLM, OpenAI, OpenAIModel, SpendGate};
//!
//! let model = OpenAIModel::Gpt4oMini.to_string();
//! let gate: Arc<dyn SpendGate> = Arc::new(MyGate::new(50_000)); // your SpendGate
//! let llm = BudgetedLLM::new(OpenAI::default().with_model(&model), gate, &model);
//! ```
//!
//! With the `budget` feature, llm-budget's `AgentBudget` and `ScopeGuard` implement
//! [`SpendGate`], for USD caps from per-model prices and slices of one budget for parallel
//! sub-chains (see `examples/llm_budget.rs`). Nothing from llm-budget appears in this
//! module's types.

use std::{error::Error, pin::Pin, sync::Arc};

use async_trait::async_trait;
use futures::{Stream, StreamExt};

use crate::{
    language_models::{
        box_llm_events, count_tokens, llm::LLM, options::CallOptions, LLMError, LLMEvent,
        LLMEventStream, TokenUsage,
    },
    schemas::{Message, StreamData},
};

/// Output tokens reserved per call when no `max_tokens` is set.
const DEFAULT_OUTPUT_RESERVE: u32 = 1_024;

/// Decides whether a model call may be sent, and books what it used.
pub trait SpendGate: Send + Sync {
    /// Reserves the most a call to `model` can use: `prompt_tokens` in and up to
    /// `max_output_tokens` out. An error refuses the call before it is sent; the caller
    /// gets it back as [`LLMError::SpendRefused`].
    fn reserve(
        &self,
        model: &str,
        prompt_tokens: u64,
        max_output_tokens: u64,
    ) -> Result<Box<dyn SpendHold>, Box<dyn Error + Send + Sync>>;
}

/// One reserved call, returned by [`SpendGate::reserve`].
///
/// A hold dropped without [`settle`](SpendHold::settle) or [`cancel`](SpendHold::cancel)
/// (the call was cancelled, a stream was dropped or failed midway, or no usage was
/// reported) should be booked at the reserved amount, because the provider may already be
/// billing for the call.
pub trait SpendHold: Send {
    /// The call returned and the provider reported this usage.
    fn settle(self: Box<Self>, usage: &TokenUsage);

    /// The call failed before producing anything; nothing was used.
    fn cancel(self: Box<Self>);
}

/// Wraps an [`LLM`] so every call goes through a [`SpendGate`]. See the
/// [module docs](self).
pub struct BudgetedLLM {
    llm: Box<dyn LLM>,
    gate: Arc<dyn SpendGate>,
    model: String,
    output_reserve: u32,
    max_tokens: Option<u32>,
}

impl Clone for BudgetedLLM {
    fn clone(&self) -> Self {
        Self {
            llm: self.llm.clone_box(),
            gate: Arc::clone(&self.gate),
            model: self.model.clone(),
            output_reserve: self.output_reserve,
            max_tokens: self.max_tokens,
        }
    }
}

impl BudgetedLLM {
    /// Puts `llm` behind `gate`. `model` is the id passed to the gate and used to count
    /// prompt tokens, e.g. `"gpt-4o-mini"`.
    pub fn new<L: LLM + 'static>(
        llm: L,
        gate: Arc<dyn SpendGate>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            llm: Box::new(llm),
            gate,
            model: model.into(),
            output_reserve: DEFAULT_OUTPUT_RESERVE,
            max_tokens: None,
        }
    }

    /// Output tokens to reserve per call when the call options set no `max_tokens`
    /// (default 1024).
    pub fn with_output_reserve(mut self, tokens: u32) -> Self {
        self.output_reserve = tokens;
        self
    }

    fn reserve(&self, messages: &[Message]) -> Result<Hold, LLMError> {
        let prompt = count_tokens(&self.model, &self.llm.messages_to_string(messages)) as u64;
        let output = u64::from(self.max_tokens.unwrap_or(self.output_reserve));
        let hold = self
            .gate
            .reserve(&self.model, prompt, output)
            .map_err(LLMError::SpendRefused)?;
        Ok(Hold(Some(hold)))
    }
}

/// The gate's hold for one call. Settled without usage, or dropped, it falls to the
/// gate's own drop, which books the reservation.
struct Hold(Option<Box<dyn SpendHold>>);

impl Hold {
    fn settle(mut self, tokens: Option<&TokenUsage>) {
        if let (Some(tokens), Some(hold)) = (tokens, self.0.take()) {
            hold.settle(tokens);
        }
    }

    fn cancel(mut self) {
        if let Some(hold) = self.0.take() {
            hold.cancel();
        }
    }
}

#[async_trait]
impl LLM for BudgetedLLM {
    async fn generate(
        &self,
        messages: &[Message],
    ) -> Result<crate::language_models::GenerateResult, LLMError> {
        let hold = self.reserve(messages)?;
        match self.llm.generate(messages).await {
            Ok(result) => {
                hold.settle(result.tokens.as_ref());
                Ok(result)
            }
            Err(e) => {
                hold.cancel();
                Err(e)
            }
        }
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let hold = self.reserve(messages)?;
        let mut inner = match self.llm.stream(messages).await {
            Ok(inner) => inner,
            Err(e) => {
                hold.cancel();
                return Err(e);
            }
        };
        Ok(Box::pin(async_stream::stream! {
            let mut hold = Some(hold);
            let mut started = false;
            let mut usage: Option<TokenUsage> = None;
            while let Some(chunk) = inner.next().await {
                match &chunk {
                    Ok(data) => {
                        started = true;
                        if data.tokens.is_some() {
                            usage = data.tokens.clone();
                        }
                    }
                    Err(_) if !started => {
                        if let Some(hold) = hold.take() {
                            hold.cancel();
                        }
                    }
                    Err(_) => {}
                }
                yield chunk;
            }
            if let Some(hold) = hold.take() {
                hold.settle(usage.as_ref());
            }
        }))
    }

    fn generate_stream<'a>(&'a self, messages: &'a [Message]) -> LLMEventStream<'a> {
        box_llm_events(async_stream::stream! {
            let hold = match self.reserve(messages) {
                Ok(hold) => hold,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            let mut hold = Some(hold);
            let mut started = false;
            let mut events = self.llm.generate_stream(messages);
            while let Some(event) = events.next().await {
                match &event {
                    Ok(LLMEvent::Done(result)) => {
                        if let Some(hold) = hold.take() {
                            hold.settle(result.tokens.as_ref());
                        }
                    }
                    Ok(_) => started = true,
                    Err(_) if !started => {
                        if let Some(hold) = hold.take() {
                            hold.cancel();
                        }
                    }
                    Err(_) => {}
                }
                yield event;
            }
        })
    }

    fn add_options(&mut self, options: CallOptions) {
        if options.max_tokens.is_some() {
            self.max_tokens = options.max_tokens;
        }
        self.llm.add_options(options);
    }

    fn messages_to_string(&self, messages: &[Message]) -> String {
        self.llm.messages_to_string(messages)
    }
}

/// [`SpendGate`] for llm-budget: USD caps from its dated per-model price table.
#[cfg(feature = "budget")]
mod llm_budget_gate {
    use llm_budget::{AgentBudget, Reservation, ScopeGuard, Usage};

    use super::*;

    /// A whole budget.
    impl SpendGate for AgentBudget {
        fn reserve(
            &self,
            model: &str,
            prompt_tokens: u64,
            max_output_tokens: u64,
        ) -> Result<Box<dyn SpendHold>, Box<dyn Error + Send + Sync>> {
            let reservation =
                self.reserve_for(model, &Usage::new(prompt_tokens, max_output_tokens))?;
            Ok(Box::new(BudgetHold {
                reservation: Some(reservation),
                model: model.to_string(),
            }))
        }
    }

    /// A slice of a budget from `AgentBudget::child`, for one sub-chain or sub-agent.
    /// Spend in it is booked on the parent as it settles, and the unused part goes back to
    /// the parent when the last `Arc` holding the slice is dropped.
    impl SpendGate for ScopeGuard {
        fn reserve(
            &self,
            model: &str,
            prompt_tokens: u64,
            max_output_tokens: u64,
        ) -> Result<Box<dyn SpendHold>, Box<dyn Error + Send + Sync>> {
            SpendGate::reserve(&**self.budget(), model, prompt_tokens, max_output_tokens)
        }
    }

    struct BudgetHold {
        reservation: Option<Reservation>,
        model: String,
    }

    impl SpendHold for BudgetHold {
        fn settle(mut self: Box<Self>, t: &TokenUsage) {
            if let Some(reservation) = self.reservation.take() {
                let usage = Usage {
                    input_tokens: t.prompt_tokens.into(),
                    cached_input_tokens: t.cache_read_tokens.into(),
                    cache_write_tokens: t.cache_creation_tokens.into(),
                    output_tokens: t.completion_tokens.into(),
                };
                let _ = reservation.settle_usage(&self.model, &usage);
            }
        }

        fn cancel(mut self: Box<Self>) {
            if let Some(reservation) = self.reservation.take() {
                reservation.release();
            }
        }
    }

    impl Drop for BudgetHold {
        fn drop(&mut self) {
            if let Some(reservation) = self.reservation.take() {
                let estimate = reservation.estimate_usd();
                let _ = reservation.settle(estimate);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures::StreamExt;

    use super::*;
    use crate::{
        language_models::{GenerateResult, LLMEvent},
        test_utils::FakeLLM,
    };

    const MODEL: &str = "gpt-4o-mini";

    fn answer(prompt_tokens: u32, completion_tokens: u32) -> GenerateResult {
        GenerateResult {
            generation: "ok".into(),
            tokens: Some(TokenUsage::new(prompt_tokens, completion_tokens)),
            ..Default::default()
        }
    }

    /// A token-count gate with no pricing at all: `limit` tokens in total.
    #[derive(Default)]
    struct TokenGate {
        limit: u64,
        state: Arc<Mutex<TokenState>>,
    }

    #[derive(Default)]
    struct TokenState {
        reserved: u64,
        used: u64,
    }

    struct TokenHold {
        amount: u64,
        state: Arc<Mutex<TokenState>>,
        done: bool,
    }

    impl TokenGate {
        fn new(limit: u64) -> Arc<Self> {
            Arc::new(Self {
                limit,
                ..Default::default()
            })
        }
        fn reserved(&self) -> u64 {
            self.state.lock().unwrap().reserved
        }
        fn used(&self) -> u64 {
            self.state.lock().unwrap().used
        }
    }

    impl SpendGate for TokenGate {
        fn reserve(
            &self,
            _model: &str,
            prompt_tokens: u64,
            max_output_tokens: u64,
        ) -> Result<Box<dyn SpendHold>, Box<dyn Error + Send + Sync>> {
            let amount = prompt_tokens + max_output_tokens;
            let mut s = self.state.lock().unwrap();
            if s.used + s.reserved + amount > self.limit {
                return Err("token limit".into());
            }
            s.reserved += amount;
            Ok(Box::new(TokenHold {
                amount,
                state: Arc::clone(&self.state),
                done: false,
            }))
        }
    }

    impl SpendHold for TokenHold {
        fn settle(mut self: Box<Self>, usage: &TokenUsage) {
            let mut s = self.state.lock().unwrap();
            s.reserved -= self.amount;
            s.used += u64::from(usage.prompt_tokens + usage.completion_tokens);
            self.done = true;
        }
        fn cancel(mut self: Box<Self>) {
            self.state.lock().unwrap().reserved -= self.amount;
            self.done = true;
        }
    }

    impl Drop for TokenHold {
        fn drop(&mut self) {
            if !self.done {
                let mut s = self.state.lock().unwrap();
                s.reserved -= self.amount;
                s.used += self.amount;
            }
        }
    }

    #[tokio::test]
    async fn books_the_reported_usage_and_frees_the_hold() {
        let gate = TokenGate::new(10_000);
        let llm = BudgetedLLM::new(
            FakeLLM::with_results([answer(1_000, 200)]),
            gate.clone(),
            MODEL,
        );
        llm.invoke("hello").await.unwrap();
        assert_eq!(gate.used(), 1_200);
        assert_eq!(gate.reserved(), 0);
    }

    #[tokio::test]
    async fn refuses_a_call_that_does_not_fit_before_sending_it() {
        let fake = FakeLLM::with_results([answer(1, 1)]);
        // Less than the default 1024-token output reservation.
        let llm = BudgetedLLM::new(fake.clone(), TokenGate::new(500), MODEL);
        let err = llm.invoke("hello").await.unwrap_err();
        assert!(matches!(err, LLMError::SpendRefused(_)), "{err}");
        assert!(fake.calls().is_empty());
    }

    #[tokio::test]
    async fn max_tokens_sets_the_output_reservation() {
        let fake = FakeLLM::with_results([answer(1, 1)]);
        let mut llm = BudgetedLLM::new(fake.clone(), TokenGate::new(500), MODEL);
        llm.add_options(CallOptions::new().with_max_tokens(16));
        llm.invoke("hello").await.unwrap();
        assert_eq!(fake.calls().len(), 1);
    }

    #[tokio::test]
    async fn a_failed_call_books_nothing() {
        let gate = TokenGate::new(10_000);
        let llm = BudgetedLLM::new(FakeLLM::new(Vec::<String>::new()), gate.clone(), MODEL);
        assert!(llm.invoke("hello").await.is_err());
        assert_eq!(gate.used(), 0);
        assert_eq!(gate.reserved(), 0);
    }

    #[tokio::test]
    async fn generate_stream_settles_on_done() {
        let gate = TokenGate::new(10_000);
        let llm = BudgetedLLM::new(
            FakeLLM::with_results([answer(500, 50)]),
            gate.clone(),
            MODEL,
        );
        let messages = [Message::new_human_message("hello")];
        let events: Vec<_> = llm.generate_stream(&messages).collect().await;
        assert!(matches!(events.last(), Some(Ok(LLMEvent::Done(_)))));
        assert_eq!(gate.used(), 550);
        assert_eq!(gate.reserved(), 0);
    }

    #[tokio::test]
    async fn a_stream_without_usage_books_the_reservation() {
        let gate = TokenGate::new(10_000);
        // FakeLLM's `stream` reports no token usage.
        let llm = BudgetedLLM::new(FakeLLM::new(["a few words"]), gate.clone(), MODEL)
            .with_output_reserve(100);
        let stream = llm
            .stream(&[Message::new_human_message("hello")])
            .await
            .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 3);
        assert!(gate.used() > 100);
        assert_eq!(gate.reserved(), 0);
    }

    #[cfg(feature = "budget")]
    mod llm_budget {
        use ::llm_budget::{AgentBudget, Limit, Usage};

        use super::*;

        fn budget(usd: f64) -> Arc<AgentBudget> {
            Arc::new(AgentBudget::builder("test", usd).build().unwrap())
        }

        fn cost(prompt_tokens: u64, completion_tokens: u64) -> f64 {
            budget(1.0)
                .cost_of(MODEL, &Usage::new(prompt_tokens, completion_tokens))
                .unwrap()
        }

        #[tokio::test]
        async fn books_usd_at_the_model_price() {
            let budget = budget(1.0);
            let llm = BudgetedLLM::new(
                FakeLLM::with_results([answer(1_000, 200)]),
                budget.clone(),
                MODEL,
            );
            llm.invoke("hello").await.unwrap();
            assert!((budget.spent_usd() - cost(1_000, 200)).abs() < 1e-12);
            assert_eq!(budget.reserved_usd(), 0.0);
        }

        #[tokio::test]
        async fn refuses_with_the_budget_error() {
            let fake = FakeLLM::with_results([answer(1, 1)]);
            // 1024 output tokens of gpt-4o-mini cost more than this.
            let llm = BudgetedLLM::new(fake.clone(), budget(0.000_1), MODEL);
            let LLMError::SpendRefused(e) = llm.invoke("hello").await.unwrap_err() else {
                panic!("expected SpendRefused");
            };
            assert!(e.downcast_ref::<::llm_budget::BudgetError>().is_some());
            assert!(fake.calls().is_empty());
        }

        #[tokio::test]
        async fn slices_book_on_the_parent_and_refund_on_drop() {
            let parent = budget(1.0);
            let fake = FakeLLM::with_results([answer(1_000, 200)]);
            let slice = Arc::new(parent.child("branch", Limit::Usd(0.4)).unwrap());
            assert!((parent.remaining_usd() - 0.6).abs() < 1e-9);
            // A second slice that does not fit in the $0.60 left is refused up front.
            assert!(parent.child("too-big", Limit::Usd(0.7)).is_err());

            let branch = BudgetedLLM::new(fake, slice, MODEL);
            branch.invoke("hello").await.unwrap();
            assert!((parent.spent_usd() - cost(1_000, 200)).abs() < 1e-12);
            drop(branch);
            assert!((parent.remaining_usd() - (1.0 - cost(1_000, 200))).abs() < 1e-9);
        }

        #[tokio::test]
        async fn a_slice_caps_its_branch_without_stopping_the_parent() {
            let parent = budget(1.0);
            let fake = FakeLLM::with_results([answer(1, 1), answer(1, 1)]);
            let slice = Arc::new(parent.child("branch", Limit::Usd(0.000_1)).unwrap());
            let branch = BudgetedLLM::new(fake.clone(), slice, MODEL);
            assert!(matches!(
                branch.invoke("hello").await,
                Err(LLMError::SpendRefused(_))
            ));
            BudgetedLLM::new(fake.clone(), parent, MODEL)
                .invoke("hello")
                .await
                .unwrap();
            assert_eq!(fake.calls().len(), 1);
        }
    }
}
