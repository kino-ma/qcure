use qcure::token::{Code};
use qcure::parse::{Program};

fn main() -> Result<(), Box<dyn std::error::Error>>{
    let src = "";
    let code = Code::from(src)?;
    Program::new(code)?;
    Ok(())
}
