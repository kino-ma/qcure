#[derive(Debug)]
pub struct Code();

#[derive(Debug)]
pub struct Token();

type Result<T> = std::result::Result<T, TokenizeError>;

#[derive(Debug)]
pub enum TokenizeError {}

impl Code {
    pub fn from(code: &str) -> Result<Self> {
        return Ok(Self());
    }
}

impl std::fmt::Display for TokenizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(f, "")
    }

}

impl std::error::Error for TokenizeError {

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_code() -> Result<()> {
        let code = "";
        Code::from(code)?;

        Ok(())
    }
}