pub struct Code();

pub struct Token();

#[derive(Debug)]
pub enum TokenizeError {}

impl Code {
    pub fn from(code: &str) -> Result<Self, TokenizeError> {
        return Ok(Self());
    }
}

impl std::fmt::Display for TokenizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "")
    }

}

impl std::error::Error for TokenizeError {

}