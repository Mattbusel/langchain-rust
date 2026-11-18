// USD spend caps with `BudgetedLLM` and llm-budget: every call is reserved before it is
// sent and booked from the reported usage, and parallel sub-chains each run on their own
// slice of one budget, so together they can never spend more than it holds.
//
// OPENAI_API_KEY=... cargo run --example llm_budget --features budget
#[cfg(feature = "budget")]
use std::sync::Arc;

#[cfg(feature = "budget")]
use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    fmt_template,
    llm::{
        openai::{OpenAI, OpenAIModel},
        BudgetedLLM, SpendGate,
    },
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args, template_fstring,
};
#[cfg(feature = "budget")]
use llm_budget::{AgentBudget, Limit};

#[cfg(feature = "budget")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = OpenAIModel::Gpt4oMini.to_string();
    // The whole run may spend at most one cent.
    let budget = Arc::new(AgentBudget::builder("pipeline", 0.01).build()?);

    let chain_on = |gate: Arc<dyn SpendGate>| {
        let llm = BudgetedLLM::new(OpenAI::default().with_model(&model), gate, &model);
        LLMChainBuilder::new()
            .prompt(message_formatter![fmt_template!(
                HumanMessagePromptTemplate::new(template_fstring!(
                    "In two sentences: {topic}",
                    "topic"
                ))
            )])
            .llm(llm)
            .build()
    };

    // Two sub-chains in parallel, each on its own slice reserved up front.
    let pros = chain_on(Arc::new(budget.child("pros", Limit::Usd(0.004))?))?;
    let cons = chain_on(Arc::new(budget.child("cons", Limit::Usd(0.004))?))?;
    let (pros_answer, cons_answer) = tokio::join!(
        pros.invoke(prompt_args! { "topic" => "the benefits of Rust for AI agents" }),
        cons.invoke(prompt_args! { "topic" => "the drawbacks of Rust for AI agents" }),
    );
    println!("pros: {}\n\ncons: {}", pros_answer?, cons_answer?);

    // The slices were booked on the parent as they ran; dropping the chains returns
    // what they did not use.
    drop((pros, cons));
    println!(
        "spent ${:.6} of ${:.2}, ${:.6} left",
        budget.spent_usd(),
        budget.limit_usd(),
        budget.remaining_usd()
    );
    Ok(())
}

#[cfg(not(feature = "budget"))]
fn main() {
    println!("This example requires the 'budget' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example llm_budget --features=budget");
}
