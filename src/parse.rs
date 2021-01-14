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
        let mut it = tokens.iter()
            .filter(|t| t.k != TK::WhiteSpace && t.k != TK::Empty)
            .map(|t| t.clone());
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

        expr = Expr::new(&mut it.collect())?;

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
pub enum Expr {
    Term(Term),
    FuncApplication(FuncApplicationOp),
    Bind,
}

impl Expr {
    pub fn new(v: &mut Vec<&Token>) -> Result<Self> {
        let expr = Self::term(v)
            .or(Self::func_app(v))?;
            // .or(Self::block(v));
            // ...

        Ok(expr)
    }

    pub fn term(v: &mut Vec<&Token>) -> Result<Self> {
        Term::new(v)
            .map(Self::Term)
    }

    pub fn func_app(v: &mut Vec<&Token>) -> Result<Self> {
        FuncApplication::new(v)
            .map(Self::FuncApplicationOp)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum FuncApplicationOp {
    FuncApplication(FuncApplication),
    UnaryOp { op: UnaryOp, arg: Box<FuncApplication> },
    BinaryOpL { op: BinaryOpL, arg1: Box<FuncApplicationOp>, arg2: Box<FuncApplicationOp> },
    BinaryOpR { op: BinaryOpR, arg1: Box<FuncApplicationOp>, arg2: Box<FuncApplicationOp> },
}

#[derive(Debug, PartialEq, Clone)]
pub enum FuncApplication {
    Term(Term),
    Normal { op: Box<FuncApplication>, args: Vec<Term> },
}

type UnaryOp = Term;
type BinaryOpL = Term;
type BinaryOpR = Term;

impl FuncApplication {
    pub fn new(v: &mut Vec<&Token>) -> Result<Self> {
        let first = v[0];
        
        if first.k == TK::Symbol {}
        Ok(())
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Term {
    Identifier(String),
    Operator(String),
    Literal(LiteralValue),
    Expr(Box<Expr>),
    Block { stmts: Vec<Statement>, rt_keyword: Option<()>, expr: Box<Expr> },
    Selection,
    Repetation,
}
use Term::*;

impl Term {
    pub fn new(tokens: &mut Vec<&Token>) -> Result<Self> {
        let tk = tokens[0];
        match tk.k {
            TK::Numeric => tokens.remove(0).t
                .parse::<Num>()
                .map(NumericLiteral)
                .map(Literal)
                .or(Err(InvalidNumeric)),
            TK::Identifier => Ok(Identifier(tokens.remove(0).t)),
            TK::Symbol => Ok(Operator(tokens.remove(0).t)),
            _ => Err(CouldntParse)
        }
    }

    pub fn prior(&self) -> usize {
        match self {
            Identifier(_) => 9,
            Literal(_) => 0,
            Operator(op) => match &op[..] {
                "*" | "/" | "%" => 7,
                "+" | "-" => 6,
                "&&" => 3,
                "||" => 2,
                _ => 4
            }
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

#[derive(Debug, PartialEq, Clone)]
pub struct ExprStack{
    stack: [Vec<Term>; 10],
}

impl ExprStack {
    pub fn new() -> Self {
        let stack = Self::empty_stack();
        Self::from(stack)
    }

    pub fn from(stack: [Vec<Term>; 10]) -> Self {
        Self { stack }
    }

    pub fn clear(&mut self) -> Self {
        let stack = self.stack.clone();
        self.stack = Self::empty_stack();
        Self::from(stack)
    }

    pub fn push(&mut self, term: Term) {
        self.stack[term.prior()].push(term);
    }

    pub fn pop(&mut self) -> Option<Term> {
        let mut i = 0;
        while self[i].is_empty() {
            i += 1;
        }
        self[i].pop()
    }
    
    pub fn is_empty(&self) -> bool {
        self.stack
            .iter()
            .all(|stack| stack.is_empty())
    }

    pub fn empty_stack() -> [Vec<Term>; 10] {
        [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ]
    }
}

impl<I: std::slice::SliceIndex<[Vec<Term>]>> std::ops::Index<I> for ExprStack {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.stack[index]
    }
}

impl<I: std::slice::SliceIndex<[Vec<Term>]>> std::ops::IndexMut<I> for ExprStack {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.stack[index]
    }
}

type Result<T> = std::result::Result<T, ParseError>;

#[derive(Debug, PartialEq, Clone)]
pub enum ParseError {
    UnexpectedToken(Token),
    UnexpectedCloseBracket,
    UnexpectedEOF,
    InvalidNumeric,
    CouldntParse,
}
use ParseError::*;

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ParseError::*;
        match self {
            UnexpectedToken(t) => write!(f, "unexpected token: `{:?}`", t),
            UnexpectedEOF => write!(f, "unexpected EOF"),
            UnexpectedCloseBracket => write!(f, "unexpected closing bracket")
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_program() {
        let expr_src = complex_expr_src();
        let src = format!("hoge := {}", expr_src);

        let code = Code::from(&src).expect("failed to tokenize");
        Program::new(code).expect("failed to parse");
    }

    #[test]
    fn new_statement() {
        let src = "hoge := 1";
        let code = Code::from(src).expect("failed to tokenize");

        let mut iter = code.iter();

        let expr = Expr::new(&mut vec![&Token::new("1".to_string(), TK::Numeric)]).unwrap();
        let expect = Statement::Assign { prefix: None, ident: "hoge".to_string(), expr };
        let actual = Statement::new(&mut iter).unwrap();

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_complex_expr() {
        let src = complex_expr_src();
        let tokens = Code::from(src).expect("failed to tokenize").tokens;

        let expect: Expr = complex_expr();

        let actual = Expr::new(&mut tokens.iter().collect()).expect("failed to parse expr");
        assert_eq!(expect, actual);
    }

    fn complex_expr_src() -> &'static str {
        "f 1 + 2 * (-3 + 4)"
    }

    fn complex_expr() -> Expr {
        Expr::FuncApplication (
            FuncApplicationOp::BinaryOpL {
                op: Term::Operator("+".to_string()),
                arg1: Box::new(FuncApplicationOp::FuncApplication(FuncApplication::Normal{
                        op: Box::new(FuncApplication::Term(
                            Term::Identifier("f".to_string())
                        )),
                        args: vec![Term::Literal(NumericLiteral(1))]
                    })
                ),
                arg2: Box::new(FuncApplicationOp::BinaryOpL {
                    op: Term::Operator("*".to_string()),
                    arg1: Box::new(FuncApplicationOp::FuncApplication(FuncApplication::Term(Term::Literal(LiteralValue::NumericLiteral(
                        2
                    ))))),
                    arg2: Box::new(FuncApplicationOp::BinaryOpL {
                        op: Term::Operator(
                            "+".to_string()
                        ),
                        arg1: Box::new(FuncApplicationOp::UnaryOp{
                            op: Term::Operator(
                                "-".to_string()
                            ),
                            arg: Box::new(FuncApplication::Term(Term::Literal(LiteralValue::NumericLiteral(
                                3
                            ))))
                        }),
                        arg2: Box::new(FuncApplicationOp::FuncApplication(FuncApplication::Term(Term::Literal(LiteralValue::NumericLiteral(
                            4
                        )))))
                    })
                })
            }
        )
        /*Expr(vec![
            Literal(NumericLiteral(1)),
            Identifier("f".to_string()),
            Literal(NumericLiteral(2)),
            Literal(NumericLiteral(3)),
            Identifier("-".to_string()),
            Literal(NumericLiteral(4)),
            Identifier("+".to_string()),
            Identifier("*".to_string()),
            Identifier("+".to_string()),
        ])*/
    }
}