use crate::token::Token;

pub struct Program {
    ptree: Tree
}

type Tree = Box<Node>;

pub struct Node {
    kind: Kind,
    token: Token,
}

pub enum Kind {

}