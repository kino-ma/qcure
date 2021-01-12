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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_program() {
        let code = r#"hoge fuga
        123piyo  a"#;
        let code = Code::from(code);
        Program::new(code).expect("failed to parse");
    }
}