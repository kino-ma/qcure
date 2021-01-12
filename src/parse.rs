use crate::token::{Code, Token, TokenKind::*};

pub struct Program {
    ptree: PTree
}

impl Program {
    pub fn new(code: Code) -> Result<Self> {
        let children = code.tokens;
        let kind = Kind::Empty;
        let (token, _) = Token::empty();
        let tree = Tree::Leaf(token);
        let ptree = PTree { kind, tree };
        Ok(Self { ptree })
    }
}

pub struct PTree {
    kind: Kind,
    tree: Tree,
}

pub enum Tree {
    Node(Vec<PTree>),
    Leaf(Token),
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
    Empty,
}

type Result<T> = std::result::Result<T, ParseError>;

#[derive(Debug, PartialEq, Clone)]
pub enum ParseError {
    InvalidToken(Token, Token),
    Other
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ParseError::*;
        match self {
            InvalidToken(t1, t2) => write!(f, "invalid token: `{:?}` (appears after `{:?}`)", t1, t2),
            Other => write!(f, "some error"),
        }
    }
}

impl std::error::Error for ParseError {}

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

    #[test]
    fn new_node() {
        let 
    }
}