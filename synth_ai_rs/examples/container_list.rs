use synth_ai::Synth;

#[tokio::main]
async fn main() -> Result<(), synth_ai::Error> {
    let synth = Synth::from_env()?;
    let deployments = synth.list_container_deployments().await?;
    println!("deployments={}", deployments.len());
    for dep in deployments {
        println!(
            "{} {} {} {}",
            dep.deployment_id, dep.name, dep.status, dep.container_url
        );
    }
    Ok(())
}
