use qcure::token::{Code};

fn main() -> Result<(), Box<dyn std::error::Error>>{
    let code = String::from("");
    let _tokens = Code::from(&code)?;

    Ok(())
}
