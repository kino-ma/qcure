use std::collections::HashMap;
use crate::token::{Code, Token, TokenKind::*};

pub struct Program {
    stmts: Vec<Statement>,
}

impl Program {
    pub fn new(code: Code) -> Result<Self> {
        let tokens = code.tokens;
        let mut iter = tokens.iter();
        let mut stmts = Vec::new();

        // loop
        {
            let stmt = Statement::new(&mut iter);
            stmts.push(stmt);
        }

        let program = Self { stmts };
        Ok(program)
    }
}

pub enum Statement {
    Assign { prefix: Option<AssignPrefix>, ident: String, expr: Vec<Value> },
}

pub enum AssignPrefix {

}

pub enum Value {
    Identifier(String),
    Literal(LiteralValue),
}

pub enum LiteralValue {
    NumericLiteral(i64),
    CharLiteral(char),
    StringLiteral(String),
    BoolLiteral(bool),
    StructLiteral { name: String, fields: HashMap<String, LiteralValue> },
    EnumLiteral { name: String, variant: String, fields: HashMap<String, LiteralValue> },
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
    fn new_statement() {
        use Value::*;
        use LiteralValue::*;
        let src = "hoge := 1";
        let code = Code::from(src).expect("failed to tokenize");

        let mut iter = code.iter();
        let mut stmts = Vec::new();

        let expect = Statement::Assign { prefix: None, ident: "hoge".to_string(), expr: vec![Literal(NumericLiteral(1))]};
        let actual = Statement::new(&mut iter);
    }
}