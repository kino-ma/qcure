#[derive(Debug)]
pub struct Code<'a> {
    tokens: Vec<Token<'a>>
}

#[derive(Debug, PartialEq)]
pub struct Token<'a> {
    t: &'a str
}

type Result<T> = std::result::Result<T, TokenizeError>;

#[derive(Debug)]
pub enum TokenizeError {}

impl<'a> Code<'_> {
    pub fn from(code: &'a str) -> Result<Self> {
        let tokens = Vec::new();
        let res = Self {
            tokens
        };

        Ok(res)
    }
}

impl<'a> Token<'a> {
    pub fn from(t: &'a str) -> Self {
        Self {
            t
        }
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
    fn new_token() {
        let expect = Token { t: "hoge" };
        let actual = Token::from("hoge");
        assert_eq!(expect, actual);

        let expect = Token { t: &"aaa 123"[0..=2] };
        let actual = Token::from("aaa");
        assert_eq!(expect, actual);

        let expect = Token { t: "123" };
        let actual = Token::from("123");
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_code() -> Result<()> {
        let code = "";
        Code::from(code)?;

        Ok(())
    }
}