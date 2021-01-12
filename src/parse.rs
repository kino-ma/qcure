use crate::token::{Code, Token, TokenKind::*};

pub struct Program {
    ptree: Tree
}

type Tree = Box<Node>;

pub struct Node {
    //kind: Kind,
    token: Token,
}

pub enum Kind {
    Keyword,
    Identifier,
    TypeIdentifier,
    BinaryOperator,
    UnaryOperator,
    Parenthesis,
    Bracket,
    Brace,
}

type Result<T> = std::result::Result<T, ParseError>;

#[derive(Debug, PartialEq, Clone)]
pub enum ParseError {
    InvalidToken(Token, Token),
    Other
}

impl Program {
    pub fn new(code: Code) -> Result<Self> {
        let (token, _) = Token::empty();
        let n = Node { token };
        let ptree = Box::new(n);
        Ok(Self { ptree })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_program() {
        let code = r#"hoge fuga
        123piyo  a"#;
        let code = Code::from(code).expect("failed to tokenize");
        Program::new(code).expect("failed to parse");
    }
}