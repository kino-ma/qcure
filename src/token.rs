#[derive(Debug, PartialEq)]
pub struct Code<'a> {
    tokens: Vec<Token<'a>>
}

#[derive(Debug, PartialEq, Clone)]
pub struct Token<'a> {
    t: &'a str
}

type Result<T> = std::result::Result<T, TokenizeError>;

#[derive(Debug)]
pub enum TokenizeError {}

impl<'a> Code<'a> {
    pub fn from(code: &'a str) -> Result<Self> {
        let mut chrs = code.chars();
        let mut tokens = Vec::new();

        loop {
            let t = Token::from(&mut chrs);
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

impl<'a> Token<'a> {
    pub fn new(t: &'a str) -> Self {
        Self {
            t
        }
    }

    pub fn from(chrs: &mut std::str::Chars<'a>) -> Self {
        let t = Self::numeric(chrs.clone())
            .or(Self::identifier(chrs.clone()))
            .unwrap_or(Self::empty());
        let _ = chrs.skip(t.len());

        t
    }

    pub fn from_chrs(chrs: &std::str::Chars<'a>, last: usize) -> Option<Self> {
        if last == 0 {
            None
        } else {
            let s: &str = &chrs.as_str()[..last];
            Some(Self::new(s))
        }
    }

    pub fn whitespace(chrs: std::str::Chars<'a>) -> Option<Self> {
        let mut idx: usize = 0;

        for c in chrs.clone() {
            if !c.is_ascii_whitespace() {
                break;
            }
            idx += 1;
        }

        Self::from_chrs(&chrs, idx)
    }

    pub fn numeric(chrs: std::str::Chars<'a>) -> Option<Self> {
        let mut idx: usize = 0;

        for c in chrs.clone() {
            if !c.is_ascii_digit() {
                break;
            }
            idx += 1;
        }

        Self::from_chrs(&chrs, idx)
    }

    pub fn identifier(chrs: std::str::Chars<'a>) -> Option<Self> {
        let mut idx: usize = 0;

        for c in chrs.clone() {
            if idx == 0 && c.is_ascii_digit() {
                break;
            }

            if !c.is_ascii_alphanumeric() && c != '\'' {
                break;
            }

            idx += 1;
        }

        Self::from_chrs(&chrs, idx)
    }

    pub fn symbol(chrs: std::str::Chars<'a>) -> Option<Self> {
        let mut idx: usize = 0;

        for c in chrs.clone() {
            if !r#"!?@#$%^&*/\|;:+-=~'"`()[]{}"#.contains(c) {
                break;
            }

            idx += 1;
        }

        Self::from_chrs(&chrs, idx)
    }

    pub fn empty() -> Self {
        Self::new("")
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
        let expect = Token { t: "hoge" };
        let actual = Token::new("hoge");
        println!("{:?}", actual);
        assert_eq!(expect, actual);

        let expect = Token { t: &"aaa 123"[0..=2] };
        let actual = Token::new("aaa");
        assert_eq!(expect, actual);

        let expect = Token { t: "123" };
        let actual = Token::new("123");
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_whitespace() {
        let expect = Token::new("  \t");
        let actual = Token::whitespace("  \thoge".chars()).unwrap();
        assert_eq!(expect, actual);

        let expect = None;
        let actual = Token::whitespace("hoge".chars());
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_numeric() {
        let expect = Token::new("123");
        let actual = Token::numeric("123 hoge".chars()).unwrap();
        assert_eq!(expect, actual);

        let expect = None;
        let actual = Token::numeric("hoge".chars());
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_identifier() {
        let expect = Token::new("hoge123");
        let actual = Token::identifier("hoge123 hoge".chars()).unwrap();
        assert_eq!(expect, actual);

        let expect = Token::new("Hoge");
        let actual = Token::identifier("Hoge".chars()).unwrap();
        assert_eq!(expect, actual);

        let expect = None;
        let actual = Token::identifier("123hoge".chars());
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_symbol() {
        let expect = Token::new(":=");
        let actual = Token::symbol(":= hoge".chars()).unwrap();
        assert_eq!(expect, actual);

        let expect = None;
        let actual = Token::symbol("hoge123".chars());
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