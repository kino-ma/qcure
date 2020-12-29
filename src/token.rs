#[derive(Debug, PartialEq)]
pub struct Code {
    tokens: Vec<Token>
}

#[derive(Debug, PartialEq, Clone)]
pub struct Token {
    t: String
}

use std::str::Chars;
type ConsumeResult<'a> = std::result::Result<Token, &'a mut Chars<'a>>;
type Result<T> = std::result::Result<T, TokenizeError>;

#[derive(Debug)]
pub enum TokenizeError {}

impl Code {
    pub fn from(code: &str) -> Result<Self> {
        let mut chars: Chars = code.chars();
        let mut tokens = Vec::new();

        loop {
            let mut chars_ = chars.clone();
            let t = Token::consume(&mut chars_);
            chars = chars_.clone();
            tokens.push(t.clone());

            if t.is_empty() {
                break
            }
        }

        let res = Self {
            tokens
        };

        Ok(res)
    }
}

impl Token {
    pub fn new(t: String) -> Self {
        let t = t.to_string();

        Self {
            t
        }
    }

    pub fn consume<'a>(chars: &'a mut Chars<'a>) -> Self {
        Self::whitespace(chars)
            .or_else(Self::numeric)
            .or_else(Self::identifier)
            .or_else(Self::symbol)
            .unwrap_or(Self::empty())
    }

    pub fn try_consume<'a>(chars: &'a mut Chars<'a>, last: usize) -> ConsumeResult<'a> {
        if last == 0 {
            Err(chars)
        } else {
            let s = chars.as_str()[..last].to_string();
            Ok(Self::new(s))
        }
    }

    pub fn whitespace<'a>(chars: &'a mut Chars<'a>) -> ConsumeResult<'a> {
        let mut idx: usize = 0;

        for c in chars.clone() {
            if !c.is_ascii_whitespace() {
                break;
            }
            idx += 1;
        }

        Self::try_consume(chars, idx)
    }

    pub fn numeric<'a>(chars: &'a mut Chars<'a>) -> ConsumeResult<'a> {
        let mut idx: usize = 0;

        for c in chars.clone() {
            if !c.is_ascii_digit() {
                break;
            }
            idx += 1;
        }

        Self::try_consume(chars, idx)
    }

    pub fn identifier<'a>(chars: &'a mut Chars<'a>) -> ConsumeResult<'a> {
        let mut idx: usize = 0;

        for c in chars.clone() {
            if idx == 0 && c.is_ascii_digit() {
                break;
            }

            if !c.is_ascii_alphanumeric() && c != '\'' {
                break;
            }

            idx += 1;
        }

        Self::try_consume(chars, idx)
    }

    pub fn symbol<'a>(chars: &'a mut Chars<'a>) -> ConsumeResult<'a> {
        let mut idx: usize = 0;

        for c in chars.clone() {
            if !r#"!?@#$%^&*/\|;:+-=~'"`()[]{}"#.contains(c) {
                break;
            }

            idx += 1;
        }

        Self::try_consume(chars, idx)
    }

    pub fn empty() -> Self {
        Self::new(String::new())
    }

    pub fn len(&self) -> usize {
        return self.t.len();
    }

    pub fn is_empty(&self) -> bool {
        self.t == ""
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
        let expect = Token { t: "hoge".to_string() };
        let actual = Token::new("hoge");
        println!("{:?}", actual);
        assert_eq!(expect, actual);

        let expect = Token { t: "aaa 123"[0..=2].to_string() };
        let actual = Token::new("aaa");
        assert_eq!(expect, actual);

        let expect = Token { t: "123".to_string() };
        let actual = Token::new("123");
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_whitespace() {
        let expect = Token::new("  \t");
        let actual = Token::whitespace(&mut "  \thoge".chars()).unwrap();
        assert_eq!(expect, actual);

        let expect = None;
        let actual = Token::whitespace(&mut "hoge".chars());
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_numeric() {
        let expect = Token::new("123");
        let actual = Token::numeric(&mut "123 hoge".chars()).unwrap();
        assert_eq!(expect, actual);

        let expect = None;
        let actual = Token::numeric(&mut "hoge".chars());
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_identifier() {
        let expect = Token::new("hoge123");
        let actual = Token::identifier(&mut "hoge123 hoge".chars()).unwrap();
        assert_eq!(expect, actual);

        let expect = Token::new("Hoge");
        let actual = Token::identifier(&mut "Hoge".chars()).unwrap();
        assert_eq!(expect, actual);

        let expect = None;
        let actual = Token::identifier(&mut "123hoge".chars());
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_symbol() {
        let expect = Token::new(":=");
        let actual = Token::symbol(&mut ":= hoge".chars()).unwrap();
        assert_eq!(expect, actual);

        let expect = None;
        let actual = Token::symbol(&mut "hoge123".chars());
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_code() {
        let code = r#"hoge fuga
        123piyo  a"#;

        let arr = ["hoge", " ", "fuga", "\n        ", "123", "piyo", "  ", "a"];
        let tokens = arr.iter().map(|s| Token::new(s)).collect();
        let expect = Code { tokens };

        let actual = Code::from(code).unwrap();
        
        assert_eq!(expect, actual);
    }
}