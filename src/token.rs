#[derive(Debug, PartialEq)]
pub struct Code {
    pub tokens: Vec<Token>
}

#[derive(Debug, PartialEq, Clone)]
pub struct Token {
    t: String,
    k: TokenKind,
}

#[derive(Debug, PartialEq, Clone)]
pub enum TokenKind {
    WhiteSpace,
    Numeric,
    Identifier,
    Symbol,
    Empty
}
use TokenKind::*;

type ConsumeResult = std::result::Result<(Token, String), String>;
type Result<T> = std::result::Result<T, TokenizeError>;

#[derive(Debug)]
pub enum TokenizeError {}

impl Code {
    pub fn from<'a>(code: &str) -> Result<Self> {
        let mut code = code.to_string();
        let mut tokens = Vec::new();

        loop {
            let (t, chars) = Token::consume(&code);

            if t.is_empty() {
                break
            }
            
            tokens.push(t);
            code = chars.to_string();
        }

        let res = Self {
            tokens
        };

        Ok(res)
    }
}

impl Token {
    pub fn new(t: String, k: TokenKind) -> Self {
        let t = t.to_string();

        Self {
            t, k
        }
    }

    pub fn consume(s: &str) -> (Self, String) {
        let s = s.to_string();
        Self::whitespace(s)
            .or_else(Self::numeric)
            .or_else(Self::identifier)
            .or_else(Self::symbol)
            .unwrap_or(Self::empty())
    }

    pub fn try_consume(code: String, k: TokenKind, last: usize) -> ConsumeResult {
        if last == 0 {
            Err(code)
        } else {
            let s = code.chars().take(last).collect();
            let c = code.chars().skip(last).collect();
            Ok((Self::new(s, k), c))
        }
    }

    pub fn whitespace<'a>(code: String) -> ConsumeResult {
        let mut idx: usize = 0;

        let mut chars = code.chars();
        while let Some(c) =  chars.next() {
            if !c.is_ascii_whitespace() {
                break;
            }
            idx += 1;
        }

        Self::try_consume(code, WhiteSpace, idx)
    }

    pub fn numeric<'a>(code: String) -> ConsumeResult {
        let mut idx: usize = 0;

        for c in code.chars() {
            if !c.is_ascii_digit() {
                break;
            }
            idx += 1;
        }

        Self::try_consume(code, Numeric, idx)
    }

    pub fn identifier<'a>(code: String) -> ConsumeResult {
        let mut idx: usize = 0;

        for c in code.chars() {
            if idx == 0 && c.is_ascii_digit() {
                break;
            }

            if !c.is_ascii_alphanumeric() && c != '\'' {
                break;
            }

            idx += 1;
        }

        Self::try_consume(code, Identifier, idx)
    }

    pub fn symbol<'a>(code: String) -> ConsumeResult {
        let mut idx: usize = 0;

        for c in code.chars() {
            if !r#"!?@#$%^&*/\|;:+-=~'"`()[]{}"#.contains(c) {
                break;
            }

            idx += 1;
        }

        Self::try_consume(code, Symbol, idx)
    }

    pub fn empty() -> (Self, String) {
        (Self::new(String::new(), Empty), String::new())
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
        let expect = Token { t: "hoge".to_string(), k: Identifier };
        let actual = Token::new("hoge".to_string(), Identifier);
        println!("{:?}", actual);
        assert_eq!(expect, actual);

        let expect = Token { t: "aaa 123"[0..=2].to_string(), k: Identifier };
        let actual = Token::new("aaa".to_string(), Identifier);
        assert_eq!(expect, actual);

        let expect = Token { t: "123".to_string(), k: Numeric };
        let actual = Token::new("123".to_string(), Numeric);
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_whitespace() {
        let expect = Token::new("  \t".to_string(), WhiteSpace);
        let actual = Token::whitespace("  \thoge".to_string()).unwrap().0;
        assert_eq!(expect, actual);

        let expect = Err("hoge".to_string());
        let actual = Token::whitespace("hoge".to_string());
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_numeric() {
        let expect = Token::new("123".to_string(), Numeric);
        let actual = Token::numeric("123 hoge".to_string()).unwrap().0;
        assert_eq!(expect, actual);

        let expect = Err("hoge".to_string());
        let actual = Token::numeric("hoge".to_string());
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_identifier() {
        let expect = Token::new("hoge123".to_string(), Identifier);
        let actual = Token::identifier("hoge123 hoge".to_string()).unwrap().0;
        assert_eq!(expect, actual);

        let expect = Token::new("Hoge".to_string(), Identifier);
        let actual = Token::identifier("Hoge".to_string()).unwrap().0;
        assert_eq!(expect, actual);

        let expect = Err("123hoge".to_string());
        let actual = Token::identifier("123hoge".to_string());
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_symbol() {
        let expect = Token::new(":=".to_string(), Symbol);
        let actual = Token::symbol(":= hoge".to_string()).unwrap().0;
        assert_eq!(expect, actual);

        let expect = Err("hoge123".to_string());
        let actual = Token::symbol("hoge123".to_string());
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_code() {
        let code = r#"hoge fuga
        123piyo  a"#;

        let arr = [("hoge", Identifier), (" ", WhiteSpace), ("fuga", Identifier), ("\n        ", WhiteSpace), ("123", Numeric), ("piyo", Identifier), ("  ", WhiteSpace), ("a", Identifier)];
        let tokens = arr.iter().map(|(s, k)| Token::new(s.to_string(), k.clone())).collect();
        let expect = Code { tokens };

        let actual = Code::from(code).unwrap();
        
        assert_eq!(expect, actual);
    }

    #[test]
    fn iter_tokens() {
        let code = r#"hoge fuga
        123piyo  a"#;

        let arr = [("hoge", Identifier), (" ", WhiteSpace), ("fuga", Identifier), ("\n        ", WhiteSpace), ("123", Numeric), ("piyo", Identifier), ("  ", WhiteSpace), ("a", Identifier)];
        let tokens: Vec<Token> = arr.iter().map(|(s, k)| Token::new(s.to_string(), k.clone())).collect();

        let expect = tokens.clone();

        let code = Code { tokens };
        let iter = code.iter();
        let actual: Vec<Token> = iter.collect();

        assert_eq!(expect, actual);
    }
}