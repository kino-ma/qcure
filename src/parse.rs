use std::collections::HashMap;
use crate::token::{Code, Token, TokenKind, TokenIter};

pub struct Program {
    stmts: Vec<Statement>,
}

impl Program {
    pub fn new(code: Code) -> Result<Self> {
        let mut iter = code.iter();
        let mut stmts = Vec::new();

        // loop
        {
            let stmt = Statement::new(&mut iter)?;
            stmts.push(stmt);
        }

        let program = Self { stmts };
        Ok(program)
    }
}

fn expect<'a>(it: &mut std::slice::Iter<&'a Token>, kind: Option<TokenKind>, s: Option<&str>) -> Result<&'a Token> {
    let t = it.next().ok_or(UnexpectedEOF)?;

    if let Some(kind) = kind {
        if t.k != kind {
            return Err(UnexpectedToken(*t.clone()));
        }
    }
    
    if let Some(s) = s {
        if !t.is(s) {
            return Err(UnexpectedToken(*t.clone()));
        }
    }

    Ok(t)
}

pub enum Statement {
    Assign { prefix: Option<AssignPrefix>, ident: String, expr: Expr },
}

impl Statement {
    pub fn new(iter: &mut TokenIter) -> Result<Self> {
        let mut tokens = Vec::new();

        for t in iter {
            if t.is(";") {
                break;
            }

            tokens.push(t);
        }

        Self::assign(&tokens)
            // .or(Self::definition(&tokens))
            // .or(Self::type_assertion(&tokens))
    }

    pub fn assign(tokens: &Vec<&Token>) -> Result<Self> {
        let mut it = tokens.iter();
        let mut t;

        let prefix;
        let ident;
        let expr;

        t = expect(&mut it, Some(Identifier), None)?;

        prefix = if t.is("public") || t.is("exported") {
            t = expect(&mut it, Some(Identifier), None)?;
            None
        } else {
            None
        };

        ident = t.t.clone();

        expect(&mut it, Some(Symbol), Some(":="));

        expr = Expr::new(it.collect());

        Ok(Self::Assign{
            prefix,
            ident,
            expr
        })
    }
}

pub enum AssignPrefix {

}

#[derive(Debug, PartialEq, Clone)]
pub struct Expr(Vec<Term>);

#[derive(Debug, PartialEq, Clone)]
pub enum Term {
    Identifier(String),
    Literal(LiteralValue),
}
use Term::*;

#[derive(Debug, PartialEq, Clone)]
pub enum LiteralValue {
    NumericLiteral(i64),
    CharLiteral(char),
    StringLiteral(String),
    BoolLiteral(bool),
    StructLiteral { name: String, fields: HashMap<String, LiteralValue> },
    EnumLiteral { name: String, variant: String, fields: HashMap<String, LiteralValue> },
}
use LiteralValue::*;

type Result<T> = std::result::Result<T, ParseError>;

#[derive(Debug, PartialEq, Clone)]
pub enum ParseError {
    UnexpectedToken(Token),
    UnexpectedEOF,
}
use ParseError::*;

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ParseError::*;
        match self {
            InvalidToken(t1, t2) => write!(f, "invalid token: `{:?}` (appears after `{:?}`)", t1, t2),
            UnexpectedEOF => write!(f, "unexpected EOF"),
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
        let src = "hoge := 1";
        let code = Code::from(src).expect("failed to tokenize");

        let mut iter = code.iter();
        let mut stmts = Vec::new();

        let expect = Statement::Assign { prefix: None, ident: "hoge".to_string(), expr: vec![Literal(NumericLiteral(1))]};
        let actual = Statement::new(&mut iter);
    }

    #[test]
    fn new_complex_expr() {
        let src = "f 1 + 2 * (-3 + 4)";
        let tokens = Code::from(src).expect("failed to tokenize").tokens;

        let expect = Expr(vec![
            Literal(NumericLiteral(1)),
            Identifier("f".to_string()),
            Literal(NumericLiteral(2)),
            Literal(NumericLiteral(3)),
            Identifier("-".to_string()),
            Literal(NumericLiteral(4)),
            Identifier("+".to_string()),
            Identifier("*".to_string()),
            Identifier("+".to_string()),
        ]);

        let actual = Expr::new(tokens);
        assert_eq!(expect, actual);
    }
}