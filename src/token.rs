#[derive(Debug)]
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
            let s: &str = &chrs.as_str()[..=last];
            Some(Self::new(s))
        }
    }

    pub fn numeric(mut chrs: std::str::Chars<'a>) -> Option<Self> {
        let mut idx: usize = 0;

        for c in chrs.next() {
            idx += 1;
            if !c.is_ascii_digit() {
                break;
            }
        }

        Self::from_chrs(&chrs, idx)
    }

    pub fn identifier(mut chrs: std::str::Chars<'a>) -> Option<Self> {
        let mut idx: usize = 0;

        for c in chrs.next() {
            if !c.is_alphanumeric() && c != '\'' {
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
    fn tokenize_numeric() {
        let expect = Token::new("123");
        let actual = Token::numeric("123 hoge".chars()).unwrap();
        assert_eq!(expect, actual);

        let expect = Token { t : "" };
        let actual = Token::numeric("hoge".chars()).unwrap();
        assert_eq!(expect, actual);
    }

    #[test]
    fn tokenize_code() -> Result<()> {
        let code = "";
        Code::from(code)?;

        Ok(())
    }
}