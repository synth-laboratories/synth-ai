use synth_ai::SynthClient;

#[tokio::main]
async fn main() -> Result<(), synth_ai::Error> {
    let client = SynthClient::from_env()?;
    let deployments = client.containers().list().await?;
    for dep in deployments.items.into_iter().chain(deployments.data.into_iter()) {
        println!(
            "container_id={:?} status={:?} url={:?}",
            dep.container_id, dep.status, dep.container_url
        );
    }
    Ok(())
}
