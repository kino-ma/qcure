use std::collections::HashMap;
use crate::token::{Code, Token, TokenKind as TK, TokenIter};

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Program {
    stmts: Vec<Statement>,
}

impl Program {
    pub fn new(code: Code) -> Result<Self> {
        let mut v = code.tokens.iter()
            .filter(|t| t.k != TK::WhiteSpace && t.k != TK::Empty)
            .map(|t| t)
            .collect();
        let mut stmts = Vec::new();

        // loop
        {
            let stmt = Statement::new(&mut v)?;
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
    match &tk.k {
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
    pub fn new(v: &mut Vec<&Token>) -> Result<Self> {
        let p = v.iter()
            .position(|t| t.is(";"))
            .unwrap_or(v.len() - 1);

        let mut tokens: Vec<&Token> = v.drain(..=p).collect();
        tokens.pop();

        Self::assign(&tokens)
            // .or(Self::definition(&tokens))
            // .or(Self::type_assertion(&tokens))
    }

    pub fn assign(tokens: &Vec<&Token>) -> Result<Self> {
        let mut it = tokens.iter()
            .map(|t| *t);
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
        if expect(&mut v.iter().map(|t| t.clone()), Some(TK::Symbol), None).is_ok() {
            let arg = Box::new(FuncApplication_::new(v)?);
            let op = v.remove(0).t.clone();
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
        let min_op = it.min_by_key(|t| priority(t)).ok_or(UnexpectedEOF)?.clone();
        let idx = v.iter().rev().rposition(|t| t == &min_op).unwrap();

        // assert poped item is symbol that we found above
        assert_eq!(v.pop().unwrap(), min_op);

        let mut v2 = v.split_off(idx - 1);
        let lhs = FuncApplicationOp_::new(v)?;
        let rhs = FuncApplicationOp_::new(&mut v2)?;

        let op = min_op.t.clone();
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
        println!("FuncApplication_::new()");
        let first = Term_::new(v)?;
        v.reverse();

        let mut args = Vec::new();
        while let Some(tk) = v.pop() {
            match Term_::from(tk) {
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
        println!("Term_::new()");
        let tk = tokens[0];
        match tk.k {
            TK::Numeric | TK::Identifier => Self::from(tokens.remove(0)),
            _ => Err(CouldntParse)
        }
    }

    pub fn from(tk: &Token) -> Result<Self> {
        println!("Term_::from({:?})", tk);
        match tk.k {
            TK::Numeric => tk.t
                .parse::<Num>()
                .map(NumericLiteral)
                .map(Literal)
                .or(Err(InvalidNumeric)),
            TK::Identifier => Ok(Identifier(tk.t.clone())),
            _ => Err(CouldntParse)
        }
    }

    pub fn prior(&self) -> usize {
        match self {
            Literal(_) => 10,
            Identifier(_) => 9,
            /*Operator(op) => match &op[..] {
                "*" | "/" | "%" => 7,
                "+" | "-" => 6,
                "&&" => 3,
                "||" => 2,
                _ => 4
            },*/
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
            UnexpectedCloseBracket => write!(f, "unexpected closing bracket"),
            UnexpectedEOF => write!(f, "unexpected EOF"),
            InvalidNumeric => write!(f, "invalid numeric literal"),
            CouldntParse => write!(f, "couldn't parse as such type"),
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

        let expr = Expr_::new(&mut vec![&Token::new("1".to_string(), TK::Numeric)]).unwrap();
        let expect = Statement::Assign { prefix: None, ident: "hoge".to_string(), expr };
        let actual = Statement::new(&mut code.iter().collect()).unwrap();

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_expr_1() {
        let src = "1";
        let code = Code::from(src).expect("failed to tokenize");

        let expect = Expr_::FuncApplication(FuncApplication(Term(Literal(NumericLiteral(1)))));

        let mut v = code.tokens.iter().collect();
        let actual = Expr_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_expr_ident() {
        let src = "a";
        let code = Code::from(src).expect("failed to tokenize");

        let expect = Expr_::FuncApplication(FuncApplication(Term(Identifier("a".to_string()))));

        let mut v = code.tokens.iter().collect();
        let actual = Expr_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_expr_add() {
        let src = "1 + 2";
        let code = Code::from(src).expect("failed to tokenize");

        let expect = Expr_::FuncApplication(BinaryOpL {
            op: "+".to_string(),
            arg1: Box::new(FuncApplication(Term(Literal(NumericLiteral(1))))),
            arg2: Box::new(FuncApplication(Term(Literal(NumericLiteral(2))))),
        });

        let mut v = code.tokens.iter().collect();
        let actual = Expr_::new(&mut v).expect("failed to parse");

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

    #[test]
    fn new_simple_binary_op() {
        let src = "1 + 2";
        let code = Code::from(src).expect("failed to tokenize");

        let expect = BinaryOpL {
            op: "+".to_string(),
            arg1: Box::new(FuncApplication(Term(Literal(NumericLiteral(1))))),
            arg2: Box::new(FuncApplication(Term(Literal(NumericLiteral(2))))),
        };

        let mut v = code.tokens.iter().collect();
        let actual = FuncApplicationOp_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_func_app() {
        let src = "f 1";
        let code = Code::from(src).expect("failed to tokenize");

        let expect = Normal {
            op: Box::new(Term_::Identifier("f".to_string())),
            args: vec![Literal(NumericLiteral(1))],
        };

        let mut v = code.tokens.iter().collect();
        let actual = FuncApplication_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_func_term() {
        let src = "1";
        let code = Code::from(src).expect("failed to tokenize");

        let expect = Term(Literal(NumericLiteral(1)));

        let mut v = code.tokens.iter().collect();
        let actual = FuncApplication_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_complex_func_app() {
        let src = "f 1 2 3";
        let code = Code::from(src).expect("failed to tokenize");

        let expect = Normal {
            op: Box::new(Term_::Identifier("f".to_string())),
            args: vec![Literal(NumericLiteral(1)), Literal(NumericLiteral(2)), Literal(NumericLiteral(3))],
        };

        let mut v = code.tokens.iter().collect();
        let actual = FuncApplication_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_term() {
        let src = "1";
        let code = Code::from(src).expect("failed to tokenize");

        let expect = Literal(NumericLiteral(1));

        let mut v = code.tokens.iter().collect();
        let actual = Term_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_bracket_term() {
        let src = "(1)";
        let code = Code::from(src).expect("failed to tokenize");

        let expect = Term_::Expr(Box::new(Expr_::FuncApplication(FuncApplication(Term(Literal(NumericLiteral(1)))))));

        let mut v = code.tokens.iter().collect();
        let actual = Term_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_complex_bracket_term() {
        let src = "(1 + 2)";
        let code = Code::from(src).expect("failed to tokenize");

        let expect = Term_::Expr(Box::new(Expr_::FuncApplication(BinaryOpL {
            op: "+".to_string(),
            arg1: Box::new(FuncApplication(Term(Literal(NumericLiteral(1))))),
            arg2: Box::new(FuncApplication(Term(Literal(NumericLiteral(2))))),
        })));

        let mut v = code.tokens.iter().collect();
        let actual = Term_::new(&mut v).expect("failed to parse");

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