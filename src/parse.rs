use crate::token::{Token, TokenKind::*};

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