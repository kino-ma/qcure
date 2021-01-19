use std::collections::HashMap;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::iter::Iterator;

use crate::token::{Code, Token, TokenKind as TK};

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Program {
    stmts: Vec<Statement>,
}

impl Program {
    pub fn new(code: Code) -> Result<Self> {
        debug!("Program::new({:?})", code);
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

fn expect<'a, T, I>(it: &mut I, kind: Option<TK>, s: Option<&str>) -> Result<&'a Token>
    where
        T: Debug + Borrow<&'a Token>,
        I: Iterator<Item = T> + Clone,
{
    debug!("expect(");
    debug!("\t{:?},", it.clone().collect::<Vec<_>>());
    debug!("\t{:?},", kind);
    debug!("\t{:?},", s);
    debug!(")");

    let t = it.next()
        .ok_or({
            debug!("expected some token");
            UnexpectedEOF
        })?
        .borrow() as &Token;

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
        TK::Symbol => match &tk.t[..] {
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
        debug!("Statement::new({:?})", v);
        let p = v.iter()
            .position(|t| t.is(";"))
            .unwrap_or(v.len() - 1);

        let tokens: Vec<&Token> = v.drain(..=p).collect();
        debug!("tokens: {:?}", tokens);

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

        debug!("it: {:?}", it);
        t = expect(&mut it, Some(TK::Identifier), None)?;

        prefix = if t.is("public") || t.is("exported") {
            t = expect(&mut it, Some(TK::Identifier), None)?;
            None
        } else {
            None
        };

        ident = t.t.clone();

        debug!("it: {:?}", it);
        expect(&mut it, Some(TK::Symbol), Some(":="))?;

        debug!("it: {:?}", it);
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
        debug!("Expr_::new({:?})", v);
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
        debug!("FuncApplicationOp_::new({:?})", v);
        Self::unary_op(v)
            .or(Self::binary_op_l(v))
            .or(Self::app(v))
            //.or(Self::binary_op_r(v))
    }

    pub fn app(v: &mut Vec<&Token>) -> Result<Self> {
        debug!("FuncApplicationOp_::app({:?})", v);
        FuncApplication_::new(v)
            .map(FuncApplication)
    }

    pub fn unary_op(v: &mut Vec<&Token>) -> Result<Self> {
        debug!("FuncApplicationOp_::unary_op({:?})", v);
        match expect(&mut v.iter().map(|t| *t), Some(TK::Symbol), None) {
            Ok(_) => {
                debug!("unary_op: symbol");
                if v.is_empty() {
                    debug!("empty vec with unary_op");
                }
                let op = v.remove(0).t.clone();
                let arg = Box::new(FuncApplication_::new(v)?);
                Ok(Self::UnaryOp {
                    op,
                    arg
                })
            },
            Err(UnexpectedEOF) => Err(UnexpectedEOF),
            Err(_) => {
                debug!("unary_op: err");
                Err(CouldntParse {
                    // out of bound
                    tk: v[0].clone(),
                    as_: "unary_op".to_string()
                })
            }
        }
    }

    pub fn binary_op_l(v: &mut Vec<&Token>) -> Result<Self> {
        debug!("FuncApplicationOp_::binary_op_l({:?})", v);
        let (idx, min_op) = search_bin_op_l(v).ok_or(CouldntFindOperator)?;

        let mut v2 = v.split_off(idx + 1);

        debug!("v1: {:?}, v2: {:?}", v, v2);
        // assert poped item is symbol that we found above
        assert_eq!(min_op, v.pop().unwrap());

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

fn search_bin_op_l<'a>(v: &Vec<&'a Token>) -> Option<(usize, &'a Token)> {
    debug!("search_bin_op_l({:?})", v);
    let mut bracket_count = 0;
    let mut idx = 0;
    let mut op_stack = Vec::new();

    for tk in v.iter() {
        if tk.k != TK::Symbol {
            // pass
        } else if tk.is("(") {
            bracket_count += 1;
        } else if tk.is(")") {
            if bracket_count <= 0 {
                return None;
            }

            bracket_count -= 1;
        } else if bracket_count == 0 {
            op_stack.push((idx, *tk));
        }

        idx += 1;
    }

    let res = op_stack.iter()
        .rev()
        .min_by_key(|(_, tk)| priority(tk))
        .map(|x| *x);
    debug!("searched: {:?}", res);

    res
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
        debug!("FuncApplication_::new({:?})", v);
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
        debug!("Term_::new({:?})", tokens);
        let tk = expect(&mut tokens.iter().map(|t| *t), None, None)?;
        match tk.k {
            TK::Numeric | TK::Identifier => Self::from(tokens.remove(0)),
            TK::Symbol => Self::expr(tokens),
            _ => Err(CouldntParse {
                tk: tk.clone(),
                as_: "term".to_string()
            })
        }
    }

    pub fn from(tk: &Token) -> Result<Self> {
        debug!("Term_::from({:?})", tk);
        match tk.k {
            TK::Numeric => tk.t
                .parse::<Num>()
                .map(NumericLiteral)
                .map(Literal)
                .or(Err(InvalidNumeric)),
            TK::Identifier => Ok(Identifier(tk.t.clone())),
            _ => Err(CouldntParse {
                tk: tk.clone(),
                as_: "term".to_string()
            })
        }
    }

    pub fn expr(tokens: &mut Vec<&Token>) -> Result<Self> {
        debug!("Term_::expr({:?})", tokens);
        let tk = tokens[0];
        if !tk.is("(") {
            return Err(CouldntParse {
                tk: tk.clone(),
                as_: "(".to_string()
            })
        }

        // opening bracket
        expect(&mut tokens.iter(), Some(TK::Symbol), Some("("))?;
        tokens.remove(0);

        let (idx, _) = search_correspond_closing_brackets(tokens)
            .ok_or(ExpectedCloseBracket)?;
        let mut expr_tokens: Vec<&Token> = tokens.drain(..=idx).collect();

        // closing bracket
        expr_tokens.pop();

        Ok(Expr(Box::new(Expr_::new(&mut expr_tokens)?)))
    }

    pub fn prior(&self) -> usize {
        debug!("Term_::prior({:?})", self);
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

fn search_correspond_closing_brackets<'a>(v: &Vec<&'a Token>) -> Option<(usize, &'a Token)> {
    let mut it = v.iter();
    let mut idx = 0;
    let mut count = 0;

    loop {
        match expect(&mut it, Some(TK::Symbol), Some(")")) {
            Ok(tk) => if count <= 0 {
                break Some((idx, tk))
            } else {
                count += 1;
            },
            Err(UnexpectedEOF) => break None,
            Err(UnexpectedToken(_)) => (),
            _ => break None
        }
        idx += 1;
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
    ExpectedCloseBracket,
    UnexpectedEOF,
    InvalidNumeric,
    CouldntFindOperator,
    CouldntParse { tk: Token, as_: String }
}
use ParseError::*;

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ParseError::*;
        match self {
            UnexpectedToken(t) => write!(f, "unexpected token: `{:?}`", t),
            UnexpectedCloseBracket => write!(f, "unexpected closing bracket"),
            ExpectedCloseBracket => write!(f, "expected closing bracket"),
            UnexpectedEOF => write!(f, "unexpected EOF"),
            InvalidNumeric => write!(f, "invalid numeric literal"),
            CouldntFindOperator => write!(f, "couldn't find an operator"),
            CouldntParse {tk, as_} => write!(f, "couldn't parse {:?} as {:?}", tk, as_),
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
        let code = setup(&src[..]);

        Program::new(code).expect("failed to parse");
    }

    #[test]
    fn new_statement() {
        let src = "hoge := 1";
        let code = setup(src);
        let mut v = create_vec(&code);

        let expr = Expr_::new(&mut vec![&Token::new("1".to_string(), TK::Numeric)]).unwrap();
        debug!("expr done");
        let expect = Statement::Assign { prefix: None, ident: "hoge".to_string(), expr };
        let actual = Statement::new(&mut v).unwrap();

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_expr_1() {
        let src = "1";
        let code = setup(src);
        let mut v = create_vec(&code);

        let expect = Expr_::FuncApplication(FuncApplication(Term(Literal(NumericLiteral(1)))));

        let actual = Expr_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_expr_ident() {
        let src = "a";
        let code = setup(src);
        let mut v = create_vec(&code);

        let expect = Expr_::FuncApplication(FuncApplication(Term(Identifier("a".to_string()))));

        let actual = Expr_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_expr_add() {
        let src = "1 + 2";
        let code = setup(src);
        let mut v = create_vec(&code);

        let expect = Expr_::FuncApplication(BinaryOpL {
            op: "+".to_string(),
            arg1: Box::new(FuncApplication(Term(Literal(NumericLiteral(1))))),
            arg2: Box::new(FuncApplication(Term(Literal(NumericLiteral(2))))),
        });

        let actual = Expr_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_complex_expr() {
        let src = complex_expr_src();
        let code = setup(src);
        let mut v = create_vec(&code);

        let expect: Expr_ = complex_expr();

        let actual = Expr_::new(&mut v).expect("failed to parse expr");
        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_binary_op() {
        let src = "1 + 2";
        let code = setup(src);
        let mut v = create_vec(&code);

        let expect = BinaryOpL {
            op: "+".to_string(),
            arg1: Box::new(FuncApplication(Term(Literal(NumericLiteral(1))))),
            arg2: Box::new(FuncApplication(Term(Literal(NumericLiteral(2))))),
        };

        let actual = FuncApplicationOp_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_func_app() {
        let src = "f 1";
        let code = setup(src);
        let mut v = create_vec(&code);

        let expect = Normal {
            op: Box::new(Term_::Identifier("f".to_string())),
            args: vec![Literal(NumericLiteral(1))],
        };

        let actual = FuncApplication_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_func_term() {
        let src = "1";
        let code = setup(src);
        let mut v = create_vec(&code);

        let expect = Term(Literal(NumericLiteral(1)));

        let actual = FuncApplication_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_complex_func_app() {
        let src = "f 1 2 3";
        let code = setup(src);
        let mut v = create_vec(&code);

        let expect = Normal {
            op: Box::new(Term_::Identifier("f".to_string())),
            args: vec![Literal(NumericLiteral(1)), Literal(NumericLiteral(2)), Literal(NumericLiteral(3))],
        };

        let actual = FuncApplication_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_term() {
        let src = "1";
        let code = setup(src);
        let mut v = create_vec(&code);

        let expect = Literal(NumericLiteral(1));

        let actual = Term_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_simple_bracket_term() {
        let src = "(1)";
        let code = setup(src);
        let mut v = create_vec(&code);

        let expect = Term_::Expr(Box::new(Expr_::FuncApplication(FuncApplication(Term(Literal(NumericLiteral(1)))))));

        let actual = Term_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn new_complex_bracket_term() {
        let src = "(1 + 2)";
        let code = setup(src);
        let mut v = create_vec(&code);

        let expect = Term_::Expr(Box::new(Expr_::FuncApplication(BinaryOpL {
            op: "+".to_string(),
            arg1: Box::new(FuncApplication(Term(Literal(NumericLiteral(1))))),
            arg2: Box::new(FuncApplication(Term(Literal(NumericLiteral(2))))),
        })));

        let actual = Term_::new(&mut v).expect("failed to parse");

        assert_eq!(expect, actual);
    }

    #[test]
    fn should_search_bin_op_l() {
        let src = "f 1 2 + 3 * (4 - 5)";
        let code = setup(src);
        let v = create_vec(&code);

        let expect_idx = 3;
        let expect_tk = &Token {
            t: "+".to_string(),
            k: TK::Symbol
        };
        let expect = (expect_idx, expect_tk);

        let actual = search_bin_op_l(&v).expect("failed to search");

        assert_eq!(expect, actual);
    }

    #[test]
    fn should_search_close_bracket() {
        let src = "f 1 (2 + 3 * (4 - 5) / 6)";
        let code = setup(src);
        let v = create_vec(&code);

        let expect_idx = 14;
        let expect_tk = &Token {
            t: ")".to_string(),
            k: TK::Symbol
        };
        let expect = (expect_idx, expect_tk);

        let actual = search_correspond_closing_brackets(&v).expect("failed to search");

        assert_eq!(expect, actual);
    }

    fn setup(src: &str) -> Code {
        let _ = env_logger::builder().is_test(true).try_init();

        Code::from(src).expect("failed to tokenize")
    }

    fn create_vec(code: &Code) -> Vec<&Token> {
        code.iter().filter(|tk| tk.k != TK::WhiteSpace && tk.k != TK::Empty).collect()
    }

    fn complex_expr_src() -> &'static str {
        "f 1 + 2 * (3 + 4)"
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