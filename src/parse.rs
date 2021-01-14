use std::collections::HashMap;
use crate::token::{Code, Token, TokenKind as TK, TokenIter};

#[derive(Debug, Eq, PartialEq, Clone)]
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

fn priority(tk: &Token) -> usize {
    match tk.k {
        Symbol => match &tk.t[..] {
            "*" | "/" | "%" => 7,
            "+" | "-" => 6,
            "&&" => 3,
            "||" => 2,
            _ => 4
        },
        TK::Identifier => 9,
        _ => 10
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum Statement {
    Assign { prefix: Option<AssignPrefix>, ident: String, expr: Expr_ },
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

        expr = Expr_::new(&mut it.collect())?;

        Ok(Self::Assign{
            prefix,
            ident,
            expr
        })
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum AssignPrefix {

}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum Expr_ {
    FuncApplication(FuncApplicationOp_),
    Bind,
}

impl Expr_ {
    pub fn new(v: &mut Vec<&Token>) -> Result<Self> {
        let expr = Self::func_app(v)?;
            // .or(Self::bind(v));
            // ...

        Ok(expr)
    }

    pub fn func_app(v: &mut Vec<&Token>) -> Result<Self> {
        FuncApplicationOp_::new(v)
            .map(Self::FuncApplication)
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum FuncApplicationOp_ {
    FuncApplication(FuncApplication_),
    UnaryOp { op: UnaryOp_, arg: Box<FuncApplication_> },
    BinaryOpL { op: BinaryOpL_, arg1: Box<FuncApplicationOp_>, arg2: Box<FuncApplicationOp_> },
    BinaryOpR { op: BinaryOpR_, arg1: Box<FuncApplicationOp_>, arg2: Box<FuncApplicationOp_> },
}
use FuncApplicationOp_::*;

impl FuncApplicationOp_ {
    pub fn new(v: &mut Vec<&Token>) -> Result<Self> {
        Self::app(v)
            .or(Self::unary_op(v))
            .or(Self::binary_op_l(v))
            //.or(Self::binary_op_r(v))
    }

    pub fn app(v: &mut Vec<&Token>) -> Result<Self> {
        FuncApplication_::new(v)
            .map(FuncApplication)
    }

    pub fn unary_op(v: &mut Vec<&Token>) -> Result<Self> {
        if v[0].k == TK::Symbol {
            let arg = Box::new(FuncApplication_::new(v)?);
            let op = v.remove(0).t;
            Ok(Self::UnaryOp {
                op,
                arg
            })
        } else {
            Err(CouldntParse)
        }
    }

    pub fn binary_op_l(v: &mut Vec<&Token>) -> Result<Self> {
        let it = v.iter().rev();
        let min_op = it.min_by_key(|t| priority(t)).ok_or(UnexpectedEOF)?;
        let idx = v.iter().rev().rposition(|t| t == min_op).unwrap();

        // assert poped item is symbol that we found above
        assert_eq!(v.pop().unwrap(), min_op);

        let mut v2 = v.split_off(idx - 1);
        let lhs = FuncApplicationOp_::new(v)?;
        let rhs = FuncApplicationOp_::new(&mut v2)?;

        let op = min_op.t;
        let arg1 = Box::new(lhs);
        let arg2 = Box::new(rhs);

        Ok(BinaryOpL {
            op,
            arg1,
            arg2
        })
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum FuncApplication_ {
    Term(Term_),
    Normal { op: Box<Term_>, args: Vec<Term_> },
}
use FuncApplication_::*;

type UnaryOp_ = String;
type BinaryOpL_ = String;
type BinaryOpR_ = String;

impl FuncApplication_ {
    pub fn new(v: &mut Vec<&Token>) -> Result<Self> {
        let first = Term_::new(v)?;
        v.reverse();

        let mut args = Vec::new();
        while let Some(tk) = v.pop() {
            match Term_::new(tk) {
                Ok(tk) => args.push(tk),
                Err(_) => v.push(tk),
            }
        }
        v.reverse();
        args.reverse();

        return Ok(
            match args.len() {
                0 => Term(first),
                _ => Normal {
                    op: Box::new(first),
                    args
                }
            }
        )
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum Term_ {
    Identifier(String),
    Literal(LiteralValue),
    Expr(Box<Expr_>),
    Block { stmts: Vec<Statement>, rt_keyword: Option<()>, expr: Box<Expr_> },
    Selection,
    Repetation,
}
use Term_::*;

impl Term_ {
    pub fn new(tokens: &mut Vec<&Token>) -> Result<Self> {
        let tk = tokens[0];
        match tk.k {
            TK::Numeric => tokens.remove(0).t
                .parse::<Num>()
                .map(NumericLiteral)
                .map(Literal)
                .or(Err(InvalidNumeric)),
            TK::Identifier => Ok(Identifier(tokens.remove(0).t)),
            _ => Err(CouldntParse)
        }
    }

    pub fn from(tk: &Token) -> Result<Self> {
        match tk.k {
            TK::Numeric => tk.t
                .parse::<Num>()
                .map(NumericLiteral)
                .map(Literal)
                .or(Err(InvalidNumeric)),
            TK::Identifier => Ok(Identifier(tk.t)),
            TK::Symbol => Ok(Operator(tk.t)),
            _ => Err(CouldntParse)
        }
    }

    pub fn prior(&self) -> usize {
        match self {
            Literal(_) => 10,
            Identifier(_) => 9,
            Operator(op) => match &op[..] {
                "*" | "/" | "%" => 7,
                "+" | "-" => 6,
                "&&" => 3,
                "||" => 2,
                _ => 4
            },
            _ => 10
        }
    }
}

impl std::cmp::PartialOrd for Term_ {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.prior().cmp(&other.prior()))
    }
}

impl std::cmp::Ord for Term_ {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.prior().cmp(&other.prior())
    }
}

type Num = i64;
#[derive(Debug, Eq, PartialEq, Clone)]
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
    stack: [Vec<Term_>; 10],
}

impl ExprStack {
    pub fn new() -> Self {
        let stack = Self::empty_stack();
        Self::from(stack)
    }

    pub fn from(stack: [Vec<Term_>; 10]) -> Self {
        Self { stack }
    }

    pub fn clear(&mut self) -> Self {
        let stack = self.stack.clone();
        self.stack = Self::empty_stack();
        Self::from(stack)
    }

    pub fn push(&mut self, term: Term_) {
        self.stack[term.prior()].push(term);
    }

    pub fn pop(&mut self) -> Option<Term_> {
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

    pub fn empty_stack() -> [Vec<Term_>; 10] {
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

impl<I: std::slice::SliceIndex<[Vec<Term_>]>> std::ops::Index<I> for ExprStack {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.stack[index]
    }
}

impl<I: std::slice::SliceIndex<[Vec<Term_>]>> std::ops::IndexMut<I> for ExprStack {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.stack[index]
    }
}

type Result<T> = std::result::Result<T, ParseError>;

#[derive(Debug, Eq, PartialEq, Clone)]
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

        let expr = Expr_::new(&mut vec![&Token::new("1".to_string(), TK::Numeric)]).unwrap();
        let expect = Statement::Assign { prefix: None, ident: "hoge".to_string(), expr };
        let actual = Statement::new(&mut iter).unwrap();

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_complex_expr() {
        let src = complex_expr_src();
        let tokens = Code::from(src).expect("failed to tokenize").tokens;

        let expect: Expr_ = complex_expr();

        let actual = Expr_::new(&mut tokens.iter().collect()).expect("failed to parse expr");
        assert_eq!(expect, actual);
    }

    fn complex_expr_src() -> &'static str {
        "f 1 + 2 * (-3 + 4)"
    }

    fn complex_expr() -> Expr_ {
        Expr_::FuncApplication (
            FuncApplicationOp_::BinaryOpL {
                op: "+".to_string(),
                arg1: Box::new(FuncApplicationOp_::FuncApplication(FuncApplication_::Normal{
                        op: Box::new(Term_::Identifier("f".to_string())),
                        args: vec![Term_::Literal(NumericLiteral(1))]
                    })
                ),
                arg2: Box::new(FuncApplicationOp_::BinaryOpL {
                    op: "*".to_string(),
                    arg1: Box::new(FuncApplicationOp_::FuncApplication(FuncApplication_::Term(Term_::Literal(LiteralValue::NumericLiteral(
                        2
                    ))))),
                    arg2: Box::new(FuncApplicationOp_::BinaryOpL {
                        op: "+".to_string(),
                        arg1: Box::new(FuncApplicationOp_::UnaryOp{
                            op: "-".to_string(),
                            arg: Box::new(FuncApplication_::Term(Term_::Literal(LiteralValue::NumericLiteral(
                                3
                            ))))
                        }),
                        arg2: Box::new(FuncApplicationOp_::FuncApplication(FuncApplication_::Term(Term_::Literal(LiteralValue::NumericLiteral(
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