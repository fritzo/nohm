import os
import re
from typing import Dict, List, Optional, Tuple

DEBUG = int(os.environ.get("NOHM_DEBUG", 0))


################################################################################
# Terms
# TODO add JOIN, TOP, BOT for nondeterminism.
# TODO add TYPE, SEC, RET for types as closures.


class Term:
    """
    Terms are nodes in an interaction net graph. Each port of a term stores the
    name of its destination port and a reference to the destination term. Each
    edge's data is thus distributed between its two endpoint terms. Edges are
    oriented, but orientation cannot be determined locally.
    """

    ports: Tuple[str, ...] = ()

    def __init__(self):
        super().__init__()
        for port in self.ports:
            setattr(self, port, None)  # initially unlinked
        self.level = 0
        # TODO add a safe tag as in Asperti98

    def pop(self, port: str) -> "Port":
        assert port in self.ports
        result = getattr(self, port)
        setattr(self, port, None)
        # Typechecking requires more than merely "return result".
        if result is None:
            return result
        port, term = result
        assert isinstance(port, str)
        assert isinstance(term, Term)
        return port, term


Port = Optional[Tuple[str, "Term"]]


def safe(port: Port) -> Tuple[str, Term]:
    assert port is not None
    return port


class MUX11(Term):  # Asperti's BOHM triangle
    """Level stepping."""

    ports = "in1", "out"
    in1: Port
    out: Port


class MUX12(Term):  # aka fan-out aka dup aka delta
    """Binary sharing."""

    ports = "in1", "out1", "out2"  # Similar to HVM's PAR,DP0,DP1
    in1: Port
    out1: Port
    out2: Port

    def __call__(self, *args: Port) -> Tuple[Port, Port]:
        for port, arg in zip(self.ports, args):
            setattr(self, port, arg)
        return ("out1", self), ("out2", self)


class MUX21(Term):  # aka fan-in aka PAR aka delta
    """Binary sharing."""

    ports = "in1", "in2", "out"  # Similar to HVM's DP0,DP1,PAR
    in1: Port
    in2: Port
    out: Port

    def __call__(self, *args: Port) -> Port:
        for port, arg in zip(self.ports, args):
            setattr(self, port, arg)
        return "out", self


class LAM(Term):  # aka lambda aka abs
    """Lambda abstraction."""

    ports = "var", "body", "out"
    var: Port
    body: Port
    out: Port

    def __call__(self, *args: Port) -> Port:
        for port, arg in zip(self.ports, args):
            setattr(self, port, arg)
        return "out", self


class APP(Term):  # aka apply
    """Function application."""

    ports = "lhs", "rhs", "out"
    lhs: Port
    rhs: Port
    out: Port

    def __call__(self, *args: Port) -> Port:
        for port, arg in zip(self.ports, args):
            setattr(self, port, arg)
        return "out", self


class JOIN(Term):
    """Nondeterministic binary choice."""

    ports = "lhs", "rhs", "out"
    lhs: Port
    rhs: Port
    out: Port

    def __call__(self, *args: Port) -> Port:
        for port, arg in zip(self.ports, args):
            setattr(self, port, arg)
        return "out", self


class BOT(Term):
    """Empty choice, i.e. divergent computation."""

    ports = ("out",)
    out: Port


class TOP(Term):
    """Error, i.e. join over all terms."""

    ports = ("out",)
    out: Port


def link(lhs: Term, lhs_port: str, rhs_port: str, rhs: Term) -> None:
    """
    Create a bidirectional edge between a pair of ports on a pair of terms.
    """
    assert lhs_port in lhs.ports
    assert rhs_port in rhs.ports
    setattr(lhs, lhs_port, (rhs_port, rhs))
    setattr(rhs, rhs_port, (lhs_port, lhs))


################################################################################
# Garbage collection


def collect(port: str, term: Term) -> None:
    """
    Recursively garbage collects a term. See Asperti98a Figure 12.13 for rules.
    """
    # FIXME thee MUX rules need to check for safe fans. See Asperti98a.
    if isinstance(term, MUX21) and port != "out":
        mux = MUX11()
        mux.out = term.pop("out")
        mux.in1 = term.pop("in2" if port == "in1" else "in1")
        return

    if isinstance(term, MUX12) and port != "in1":
        mux = MUX11()
        mux.in1 = term.pop("in1")
        mux.out = term.pop("out2" if port == "out1" else "out2")
        return

    # Otherwise just propagate collection.
    for p in term.ports:
        if p != port:
            t = term.pop(p)
            if t is not None:
                collect(*t)


################################################################################
# Reduction


def reduce(port: str, root: Term) -> Tuple[str, Term]:
    """
    Reduces a term to weak head normal form.
    See Asperti98a Figure 2.22 for sharing graph reduction rules.
    """
    # Note that when porting this to C, the C code should recycle terms
    # to reduce malloc overhead. See collect() calls in this function.
    # TODO handle levels in rules.
    # TODO add remaining rules.
    # TODO decide between weak-head or full normalization.
    # TODO check for stack self-collision and convert to BOT.

    is_normal = False  # similar to HVM runtime's init
    stack = [(is_normal, port, root)]

    while stack:
        is_normal, port, term = stack.pop()

        # Normalize subterms.
        if not is_normal:
            # TODO what if isinstance(term, MUX12)? fork?
            if isinstance(term, MUX21):
                assert port != "out", port
                stack.append((True, port, term))
                stack.append((False, "out", safe(term.out)[-1]))
            elif isinstance(term, APP):
                stack.append((True, port, term))
                lhs_port, lhs = safe(term.lhs)
                stack.append((False, lhs_port, lhs))
            elif isinstance(term, LAM):
                pass
            continue

        # Henceforth subterms are weak head normalized
        if isinstance(term, APP):
            assert port == "out"
            lhs_port, lhs = safe(term.lhs)

            # term = APP lhs rhs
            # lhs = {BOT,TOP}
            # ------------------ APP-BOT, APP-TOP
            # term = lhs
            if isinstance(lhs, (BOT, TOP)):
                head_port, head = safe(term.pop("out"))
                term.pop("lhs")
                rhs_port, rhs = safe(term.pop("rhs"))
                link(head, head_port, rhs_port, rhs)
                collect("out", term)
                continue

            # term = APP lhs rhs
            # lhs = LAM var body
            # ------------------ APP-LAM aka beta
            # term = body
            # var = rhs
            if isinstance(lhs, LAM):
                assert lhs_port == "out"
                head_port, head = safe(term.pop(port))  # FIXME requires a root
                body_port, body = safe(lhs.pop("body"))
                link(head, head_port, body_port, body)
                if lhs.var is not None:
                    var_port, var = safe(lhs.pop("var"))
                    rhs_port, rhs = safe(term.pop("rhs"))
                    link(rhs, rhs_port, var_port, var)
                collect(port, term)
                stack.append((False, body_port, body))
                continue

            # term = APP lhs rhs
            # lhs = MUX21 lhs1 lhs2
            # ---------------------- APP-MUX aka APP-PAR
            # term = MUX21 app1 app2
            # rhs1, rhs2 = MUX12 rhs
            # app1 = APP lhs1 rhs1
            # app2 = APP lhs2 rhs2
            if isinstance(lhs, MUX21):
                assert lhs_port == "out"
                head_ = term.pop("out")
                rhs_ = term.pop("rhs")
                lhs1_ = lhs.pop("in1")
                lhs2_ = lhs.pop("in2")
                rhs1_, rhs2_ = MUX12()(rhs_)
                app1_ = APP()(lhs1_, rhs1_)
                app2_ = APP()(lhs2_, rhs2_)
                MUX21()(app1_, app2_, head_)
                collect(port, term)
                # FIXME Is this right? HVM doesn't push here.
                stack.append((False,) + safe(app2_))
                stack.append((False,) + safe(app1_))
                continue

            # term = APP lhs rhs
            # lhs = JOIN lhs1 lhs2
            # ---------------------- APP-JOIN
            # term = JOIN app1 app2
            # rhs1, rhs2 = MUX21 rhs
            # app1 = APP in1 rhs1
            # app2 = APP in2 rhs2
            if isinstance(lhs, JOIN):
                assert lhs_port == "out"
                raise NotImplementedError("TODO")

        # Henceforth subterms are weak head normalized
        if isinstance(term, JOIN):
            assert port == "out"
            lhs_port, lhs = safe(term.lhs)
            rhs_port, rhs = safe(term.rhs)

            # term = JOIN lhs rhs
            # lhs = TOP
            # ------------------ JOIN-LHS-TOP
            # term = TOP
            if isinstance(lhs, TOP):
                head_port, head = safe(term.pop("out"))
                top_port, top = safe(term.pop("lhs"))
                link(head, head_port, top_port, top)
                collect("out", term)
                continue

            # term = JOIN lhs rhs
            # rhs = TOP
            # ------------------ JOIN-RHS-TOP
            # term = TOP
            if isinstance(rhs, TOP):
                head_port, head = safe(term.pop("out"))
                top_port, top = safe(term.pop("rhs"))
                link(head, head_port, top_port, top)
                collect("out", term)
                continue

            # term = JOIN lhs rhs
            # lhs = BOT
            # ------------------ JOIN-LHS-BOT
            # term = rhs
            if isinstance(lhs, BOT):
                head_port, head = safe(term.pop("out"))
                term.pop("rhs")
                collect("out", term)
                link(head, head_port, rhs_port, rhs)
                continue

            # term = JOIN lhs rhs
            # rhs = BOT
            # ------------------ JOIN-RHS-BOT
            # term = lhs
            if isinstance(rhs, BOT):
                head_port, head = safe(term.pop("out"))
                term.pop("lhs")
                collect("out", term)
                link(head, head_port, lhs_port, lhs)
                continue

            # term = JOIN lhs rhs
            # lhs = MUX21 lhs1 lhs2
            # ---------------------- JOIN-LHS-MUX
            # term = MUX21 app1 app2
            # rhs1, rhs2 = MUX12 rhs
            # app1 = APP lhs1 rhs1
            # app2 = APP lhs2 rhs2
            if isinstance(lhs, MUX21):
                assert lhs_port == "out"
                raise NotImplementedError("TODO")

            # term = JOIN lhs rhs
            # rhs = MUX21 rhs1 rhs2
            # ---------------------- JOIN-RHS-MUX
            # term = MUX21 app1 app2
            # lhs1, lhs2 = MUX21 lhs
            # app1 = APP lhs1 lhs1
            # app2 = APP lhs2 lhs2
            if isinstance(rhs, MUX21):
                assert rhs_port == "out"
                raise NotImplementedError("TODO")

        if isinstance(term, MUX12):
            if port != "out":
                x_port, x = safe(term.in1)

                if isinstance(x, MUX21):
                    assert x_port == "out"
                    out1_port, out1 = safe(term.pop("out1"))
                    out2_port, out2 = safe(term.pop("out2"))
                    in1_port, in1 = safe(x.pop("in1"))
                    in2_port, in2 = safe(x.pop("in2"))

                    # out1, out2 = MUX12 x
                    # x = MUX21 in1 in2
                    # -------------------- MUX-MUX-elim aka DUP-PAR (equal)
                    # out1 = in1
                    # out2 = in2
                    if term.level == x.level:
                        link(in1, in1_port, out1_port, out1)
                        link(in2, in2_port, out2_port, out2)

                    # out1, out2 = MUX12 x
                    # x = MUX21 in1 in2
                    # -------------------- MUX-MUX-intro aka DUP-PAR (different)
                    # out1 = MUX21 x11 x12
                    # out2 = MUX21 x21 x22
                    # x11, x21 = MUX12 in1
                    # x12, x22 = MUX12 in2
                    else:
                        mux1 = MUX21()
                        mux2 = MUX21()
                        mux3 = MUX12()
                        mux4 = MUX12()
                        link(out1, out1_port, "out", mux1)
                        link(out2, out2_port, "out", mux2)
                        link(mux1, "in1", "out1", mux3)
                        link(mux1, "in2", "out1", mux4)
                        link(mux2, "in1", "out2", mux3)
                        link(mux2, "in2", "out2", mux4)
                        link(mux3, "in1", in1_port, in1)
                        link(mux4, "in1", in2_port, in2)

                    collect(port, term)
                    collect(x_port, x)
                    continue

                # out1, out2 = MUX12 x
                # x = LAM var body
                # ------------------------- MUX-LAM aka DUP-LAM
                # out1 = LAM var1 body1
                # out2 = LAM var2 body2
                # var = MUX21 var1 var2
                # body1, body2 = MUX12 body
                if isinstance(x, LAM):
                    assert x_port == "out"
                    out1_port, out1 = safe(term.pop("out1"))
                    out2_port, out2 = safe(term.pop("out2"))
                    var_port, var = safe(x.pop("var"))
                    body_port, body = safe(x.pop("body"))
                    collect(port, term)
                    collect(x_port, x)
                    var_mux = MUX21()
                    body_mux = MUX12()
                    lam1 = LAM()
                    lam2 = LAM()
                    link(var_mux, "in1", var_port, var)
                    link(body_mux, "out", body_port, body)
                    link(lam1, "var", "out1", var_mux)
                    link(lam1, "body", "in1", body_mux)
                    link(lam2, "var", "out2", var_mux)
                    link(lam2, "body", "in2", body_mux)
                    link(out1, out1_port, "out", lam1)
                    link(out2, out2_port, "out", lam2)
                    continue

    return safe(getattr(root, port))


################################################################################
# Parsing : text -> graph


re_varname = re.compile("[a-z_][a-z0-9_]*$")
re_int = re.compile("[0-9]+$")
Env = Dict[str, Tuple[str, Term]]


def parse(text: str) -> Tuple[str, Term]:
    """
    Parse a Polish notation string to create a term graph.

    :param str text: A lambda term in polish notation.
    :returns: The root term of an interaction net graph.
    :rtype: Term
    """
    text = re.sub("#.*[\n\r]", "", text)  # remove comments
    tokens = text.strip().split()
    tokens.reverse()
    env: Env = {}
    port, term = _parse(tokens, env)
    term.pop(port)
    if DEBUG:
        validate(term)
    assert not tokens, f"Extra input: {' '.join(tokens)}"
    return port, term


def _parse(tokens: List[str], env: Env) -> Tuple[str, Term]:
    token = tokens.pop()

    if re_varname.match(token):
        name = token
        assert name in env, name
        # Add a MUX12 for each occurrence; the final MUX12 will later be removed.
        mux = MUX12()
        user_port, user = env[name]
        link(mux, "in1", user_port, user)
        env[name] = "out1", mux  # save for later
        return "out2", mux  # use now

    if token in ("APP", "JOIN"):
        lhs_port, lhs = _parse(tokens, env)
        rhs_port, rhs = _parse(tokens, env)
        term = {"APP": APP, "JOIN": JOIN}[token]()
        link(term, "lhs", lhs_port, lhs)
        link(term, "rhs", rhs_port, rhs)
        return "out", term

    if token in ("BOT", "TOP"):
        term = {"BOT": BOT, "TOP": TOP}[token]()
        return "out", term

    if token == "LAM":
        name = tokens.pop()
        assert re_varname.match(name), name
        lam = LAM()
        env = env.copy()
        env[name] = "var", lam
        body_port, body = _parse(tokens, env)
        link(lam, "body", body_port, body)
        used_port, used = env[name]
        if used is lam:  # Variable was never used.
            assert used_port == "var", used_port
            lam.pop("var")
        else:  # Variable was used at least once.
            assert used_port == "out1", used_port
            assert isinstance(used, MUX12), type(used).__name__
            assert used.out1 is None, used.out1
            # Eagerly eliminate the final mux.
            user_port, user = safe(used.out2)
            link(lam, "var", user_port, user)
            collect("in1", used)
        return "out", lam

    if token == "LET":  # syntactic sugar
        name = tokens.pop()
        defn_port, defn = _parse(tokens, env)
        env = env.copy()
        env[name] = defn_port, defn
        body_port, body = _parse(tokens, env)
        used_port, used = env[name]
        if used is defn:  # defn was never used.
            collect(defn_port, defn)
        else:  # Variable was used at least once.
            assert used_port == "out", used_port
            assert isinstance(used, MUX12), type(used).__name__
            assert used.out2 is None, used.out2
            # Eagerly eliminate the final mux.
            user_port, user = safe(used.out1)
            used_port, used = safe(used.in1)
            link(used, used_port, user_port, user)
            used.pop("out1")
            used.pop("in1")
            collect("out2", used)
        return body_port, body

    if token == "LETREC":  # syntactic sugar
        name = tokens.pop()
        mux = MUX12()
        env = env.copy()
        env[name] = "out1", mux
        body_port, body = _parse(tokens, env)
        used_port, used = env[name]
        assert isinstance(used, MUX12)
        if used is mux:  # Recursion was never used.
            collect("in1", mux)
            return body_port, body
        else:  # Recursion was used at least once.
            link(body, body_port, "in1", mux)
            return "out2", mux

    # Create a Church numeral.
    if re_int.match(token):
        n = int(token)
        term = LAM()
        body = LAM()
        term.pop("out")
        link(term, "body", "out", body)
        if n == 0:
            term.pop("var")
            link(body, "body", "var", body)
            return "out", term
        f_port = "var"
        f: Term = term
        x_port = "var"
        x: Term = body
        for _ in range(n - 1):
            app = APP()
            mux = MUX12()
            link(mux, "in1", f_port, f)
            link(app, "lhs", "out1", mux)
            mux.pop("out2")
            f_port, f = "out2", mux
            x_port, x = "out", app
        app = APP()
        link(app, "out", "body", body)
        link(app, "lhs", f_port, f)
        link(app, "rhs", x_port, x)
        return "out", term

    raise ValueError(f"Unhandled token: {token}")


################################################################################
# Readback : graph -> text


def readback(port: str, term: Term) -> str:
    env: Dict[Term, str] = {}
    tokens = _readback(port, term, env)
    tokens.reverse()
    return " ".join(tokens)


def _readback(port: str, term: Term, env: Dict[Term, str]) -> List[str]:

    while isinstance(term, MUX12) and port in ("out1", "out2"):
        port, term = getattr(term, "in1")
        # TODO support sharing via LET
        # TODO support looping via LETREC
    # TODO handle MUX21

    if isinstance(term, LAM):
        if port == "out":
            # FIXME this breaks under shared LAM terms
            name = gensym(len(env))
            env[term] = name
            body = _readback(*safe(term.body), env)
            return body + [name, "LAM"]
        if port == "var":
            name = env[term]
            return [name]

    if isinstance(term, (APP, JOIN)):
        lhs = _readback(*safe(term.lhs), env)
        rhs = _readback(*safe(term.rhs), env)
        return rhs + lhs + [type(term).__name__]

    if isinstance(term, (BOT, TOP)):
        return [type(term).__name__]

    raise NotImplementedError(f"Unsupported term: {type(term).__name__}")


_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def gensym(i: int) -> str:
    """
    a,b,...,z,aa,ab,...,az,ba,bb,...,zz,aaa,...
    """
    base = len(_ALPHABET)
    size = 1
    while i >= base**size:
        i -= base**size
        size += 1
    result = ""
    for _ in range(size):
        result = _ALPHABET[i % base] + result
        i //= base
    return result


################################################################################
# Validation


def validate(root):
    """
    Validates edges in the interaction net.
    """
    terms = {root}
    pending = {root}
    while pending:
        source = pending.pop()
        for source_port in source.ports:
            assert hasattr(source, source_port)
            d = getattr(source, source_port)
            if d is None:
                if isinstance(source, LAM) and source_port == "var":
                    continue  # lambdas can have unused variables
                elif source is root and source_port == "out":
                    continue  # the root node can have an empty out port
                raise ValueError(f"Unlinked {type(source).__name__}.{source_port}")
            destin_port, destin = d
            if destin not in terms:
                terms.add(destin)
                pending.add(destin)
                assert hasattr(destin, destin_port)
                assert getattr(destin, destin_port) == (source_port, source)


__all__ = [
    "collect",
    "parse",
    "readback",
    "reduce",
    "validate",
]
