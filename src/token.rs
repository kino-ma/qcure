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

impl<'a> Code<'a> {
    pub fn from(code: &'a str) -> Result<Self> {
        let chrs = code.chars();
        let mut tokens = Vec::new();

        for c in chrs.clone() {
            match true {
                _ if c.is_ascii_digit() => {
                    let t = Token::numeric(chrs.clone());
                    tokens.push(t);
                }

                _ if c.is_ascii_lowercase() => {
                    let t = Token::identifier(chrs.clone());
                    tokens.push(t);
                }

                _ => {}
            }
        }

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

    pub fn numeric(mut chrs: std::str::Chars<'a>) -> Self {
        let mut idx: usize = 0;

        for c in chrs.next() {
            idx += 1;
            if !c.is_ascii_digit() {
                break;
            }
        }
        let s = &chrs.as_str()[..idx];

        Self::from(s)
    }

    pub fn identifier(mut chrs: std::str::Chars<'a>) -> Self {
        let mut idx: usize = 0;

        for c in chrs.next() {
            idx += 1;
            if !c.is_alphanumeric() && c != '\'' {
                break;
            }
        }
        let s = &chrs.as_str()[..idx];

        Self::from(s)
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