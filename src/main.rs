use qcure::token::{Code};

fn main() -> Result<(), Box<std::error::Error>>{
    let code = String::from("");
    let tokens = Code::from(&code)?;

    Ok(())
}
