// sfs/ast.rs — the Kaze abstract syntax tree.
//
// PLAN-KAZE-LANGUAGE §5 (M-K1). Every node carries the line it came from so
// both the checker and the VM can point at real source, and the tree holds no
// references: parsing owns its data, so a compiled module outlives the source
// string.

/// A physical unit written next to a number. Units are normalized to their SI
/// magnitude at compile time (`ms` → seconds), and the unit itself is kept in
/// the AST so the checker can refuse `meters + m/s`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    None,
    Meter,
    Mps,
    Deg,
    Sec,
    /// Whole numbers, as written: `20%` is 20, not 0.2.
    Pct,
}

impl Unit {
    pub fn label(self) -> &'static str {
        match self {
            Unit::None => "a plain number",
            Unit::Meter => "meters",
            Unit::Mps => "m/s",
            Unit::Deg => "degrees",
            Unit::Sec => "seconds",
            Unit::Pct => "%",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::And => "and",
            BinOp::Or => "or",
        }
    }

    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
        )
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    Num(f64, Unit),
    Str(String),
    Bool(bool),
    /// A parameter or `let` binding.
    Name(String),
    /// A bare telemetry field (`battery`), read through the script's sole
    /// hardware capability.
    Ambient { field: String, family: &'static str },
    /// `recv.op(args)` — a capability call. `recv` is a declared resource.
    Call {
        recv: String,
        op: String,
        args: Vec<Expr>,
        line: u32,
    },
    /// `record.field` — a field read on a map value (a host-provided record).
    /// Maps have no literal syntax in v0.1; they arrive as arguments, and this
    /// is how a script reads one.
    Field { base: String, field: String, line: u32 },
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    /// `length of x`
    Len(Box<Expr>),
    Index(Box<Expr>, Box<Expr>),
    /// `[a, b, c]`
    List(Vec<Expr>),
}

impl Expr {
    pub fn line(&self) -> u32 {
        match self {
            Expr::Call { line, .. } | Expr::Field { line, .. } => *line,
            _ => 0,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let(String, Expr, u32),
    Set(String, Expr, u32),
    /// An expression run for its effect (a capability call).
    Eval(Expr, u32),
    If {
        cond: Expr,
        then: Vec<Stmt>,
        otherwise: Vec<Stmt>,
        line: u32,
    },
    ForEach {
        var: String,
        list: Expr,
        body: Vec<Stmt>,
        line: u32,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
        line: u32,
    },
    Return(Option<Expr>, u32),
}

impl Stmt {
    pub fn line(&self) -> u32 {
        match self {
            Stmt::Let(_, _, l)
            | Stmt::Set(_, _, l)
            | Stmt::Eval(_, l)
            | Stmt::ForEach { line: l, .. }
            | Stmt::While { line: l, .. }
            | Stmt::Return(_, l) => *l,
            Stmt::If { line, .. } => *line,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ItemKind {
    /// `action` — a routine that performs work.
    Action,
    /// `check` — a routine that inspects a machine's state.
    Check,
    /// `filter` — a routine that must end in `allow` or `reject`.
    Filter,
    /// `when <condition>` — fires on every host tick whose poll is true.
    When(Expr),
    /// `every <period>` — fires on ticks where the period divides.
    Every(f64),
}

#[derive(Clone, Debug)]
pub struct Item {
    pub name: String,
    pub kind: ItemKind,
    pub params: Vec<String>,
    pub body: Vec<Stmt>,
    pub line: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Program {
    /// Declared capabilities, exactly as written.
    pub needs: Vec<String>,
    pub items: Vec<Item>,
}

impl Program {
    pub fn item(&self, name: &str) -> Option<&Item> {
        self.items.iter().find(|i| i.name == name)
    }

    pub fn items_of_kind(&self, want: &ItemKind) -> impl Iterator<Item = &Item> {
        let want_tag = kind_tag(want);
        self.items
            .iter()
            .filter(move |i| kind_tag(&i.kind) == want_tag)
    }
}

fn kind_tag(k: &ItemKind) -> u8 {
    match k {
        ItemKind::Action => 0,
        ItemKind::Check => 1,
        ItemKind::Filter => 2,
        ItemKind::When(_) => 3,
        ItemKind::Every(_) => 4,
    }
}
