use std::collections::HashMap;
use crate::token::{Code, Token, TokenKind as TK, TokenIter};

#[derive(Debug, PartialEq, Clone)]
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

fn expect<'a, I: std::iter::Iterator<Item = &'a Token>>(it: &mut I, kind: Option<TK>, s: Option<&str>) -> Result<&'a Token> {
    let t = it.next().ok_or(UnexpectedEOF)?;

    if let Some(kind) = kind {
        if t.k != kind {
            let t = t.clone().clone();
            return Err(UnexpectedToken(t));
        }
    }
    
    if let Some(s) = s {
        if !t.is(s) {
            let t = t.clone().clone();
            return Err(UnexpectedToken(t));
        }
    }

    Ok(t)
}

#[derive(Debug, PartialEq, Clone)]
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
        let mut it = tokens.iter().map(|t| *t);
        let mut t;

        let prefix;
        let ident;
        let expr;

        t = expect(&mut it, Some(TK::Identifier), None)?;

        prefix = if t.is("public") || t.is("exported") {
            t = expect(&mut it, Some(TK::Identifier), None)?;
            None
        } else {
            None
        };

        ident = t.t.clone();

        expect(&mut it, Some(TK::Symbol), Some(":="))?;

        expr = Expr::new(it.collect());

        Ok(Self::Assign{
            prefix,
            ident,
            expr
        })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum AssignPrefix {

}

#[derive(Debug, PartialEq, Clone)]
pub struct Expr(Vec<Term>);

impl Expr {
    pub fn new(v: Vec<&Token>) -> Self {
        let terms = v.iter()
            .filter_map(|t| Term::new(t))
            .collect();
        Self(terms)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Term {
    Identifier(String),
    Literal(LiteralValue),
}
use Term::*;

impl Term {
    pub fn new(tk: &Token) -> Option<Self> {
        match tk.k {
            TK::WhiteSpace => None,
            TK::Numeric => tk.t
                .parse::<Num>()
                .map(NumericLiteral)
                .map(Literal)
                .ok(),
            TK::Identifier | TK::Symbol => Some(Identifier(tk.t.clone())),
            TK::Empty => None,
        }
    }
}

type Num = i64;
#[derive(Debug, PartialEq, Clone)]
pub enum LiteralValue {
    NumericLiteral(Num),
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
            UnexpectedToken(t) => write!(f, "unexpected token: `{:?}`", t),
            UnexpectedEOF => write!(f, "unexpected EOF"),
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

        let expect = Statement::Assign { prefix: None, ident: "hoge".to_string(), expr: Expr(vec![Literal(NumericLiteral(1))])};
        let actual = Statement::new(&mut iter).unwrap();

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_complex_expr() {
        let src = "f 1 + 2 * (-3 + 4)";
        let tokens = Code::from(src).expect("failed to tokenize").tokens;

        let expect: Expr = Expr(vec![
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

        let actual = Expr::new(tokens.iter().collect());
        assert_eq!(expect, actual);
    }
}