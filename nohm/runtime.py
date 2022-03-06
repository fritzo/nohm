import os
import re
from typing import Dict, List, Optional, Tuple

DEBUG = int(os.environ.get("NOHM_DEBUG", 0))


################################################################################
# Terms
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

    def __call__(self, *args: Port) -> Tuple[str, Term]:
        for port, arg in zip(self.ports, args):
            link((port, self), arg)
        return "out", self


class MUX12(Term):  # aka fan-out aka dup aka delta
    """Binary sharing."""

    ports = "in1", "out1", "out2"  # Similar to HVM's PAR,DP0,DP1
    in1: Port
    out1: Port
    out2: Port

    def __call__(self, *args: Port) -> Tuple[Tuple[str, Term], Tuple[str, Term]]:
        for port, arg in zip(self.ports, args):
            link((port, self), arg)
        return ("out1", self), ("out2", self)


class MUX21(Term):  # aka fan-in aka PAR aka delta
    """Binary sharing."""

    ports = "in1", "in2", "out"  # Similar to HVM's DP0,DP1,PAR
    in1: Port
    in2: Port
    out: Port

    def __call__(self, *args: Port) -> Tuple[str, Term]:
        for port, arg in zip(self.ports, args):
            link((port, self), arg)
        return "out", self


class LAM(Term):  # aka lambda aka abs
    """Lambda abstraction."""

    ports = "var", "body", "out"
    var: Port
    body: Port
    out: Port

    def __call__(self, *args: Port) -> Tuple[Tuple[str, Term], Tuple[str, Term]]:
        for port, arg in zip(self.ports, args):
            link((port, self), arg)
        return ("var", self), ("out", self)


class APP(Term):  # aka apply
    """Function application."""

    ports = "lhs", "rhs", "out"
    lhs: Port
    rhs: Port
    out: Port

    def __call__(self, *args: Port) -> Tuple[str, Term]:
        for port, arg in zip(self.ports, args):
            link((port, self), arg)
        return "out", self


class JOIN(Term):
    """Nondeterministic binary choice."""

    ports = "lhs", "rhs", "out"
    lhs: Port
    rhs: Port
    out: Port

    def __call__(self, *args: Port) -> Tuple[str, Term]:
        for port, arg in zip(self.ports, args):
            link((port, self), arg)
        return "out", self


# TODO unbox atoms BOT and TOP.
class BOT(Term):
    """Empty choice, i.e. divergent computation."""

    ports = ("out",)
    out: Port


class TOP(Term):
    """Error, i.e. join over all terms."""

    ports = ("out",)
    out: Port


class MAIN(Term):
    """Main program entry point."""

    ports = ("in1",)
    in1: Port


def link(end1: Port, end2: Port) -> None:
    """
    Create a bidirectional edge between two ends.
    """
    if end1 is not None:
        port, term = end1
        assert port in term.ports
        setattr(term, port, end2)
    if end2 is not None:
        port, term = end2
        assert port in term.ports
        setattr(term, port, end1)


################################################################################
# Garbage collection


def collect(term: Term) -> None:
    """
    Recursively garbage collects a term. See Asperti98 Figure 12.13 for rules.
    """
    # FIXME these MUX rules need to check for safe fans. See Asperti98.
    if isinstance(term, MUX21) and term.out is not None:
        assert term.in1 is None or term.in2 is None
        if term.in1 is not None:
            MUX11()(term.pop("in1"), term.pop("out"))
            return
        if term.in2 is not None:
            MUX11()(term.pop("in2"), term.pop("out"))
            return

    if isinstance(term, MUX12) and term.in1 is not None:
        assert term.out1 is None or term.out2 is None
        if term.out1 is not None:
            MUX11()(term.pop("in1"), term.pop("out1"))
            return
        if term.out2 is not None:
            MUX11()(term.pop("in1"), term.pop("out2"))
            return

    # Otherwise just propagate collection.
    for port in term.ports:
        other = term.pop(port)
        if other is not None:
            other_port, other_term = other
            other_term.pop(other_port)
            collect(other_term)


################################################################################
# Reduction


def reduce(main: MAIN) -> None:
    """
    Reduces a term to weak head normal form.
    See Asperti98 Figure 2.22 for sharing graph reduction rules.
    """
    # This finds beta redexes (APP-LAM pairs) by bubble sorting wrt the partial
    # order {BOT,TOP} > {JOIN,MUX21} > APP > LAM > MUX12, including rules:
    #   APP(MUX21) -> MUX21(APP)
    #   APP(JOIN) -> JOIN(APP)
    #   MUX12(MUX21) -> wire or MUX21(MUX12)
    #   MUX12(JOIN) -> JOIN(MUX12)
    #   MUX12(LAM) -> LAM(MUX12)
    #   JOIN(TOP) -> TOP
    #   JOIN(BOT,x) -> x
    #   MUX12(TOP) -> TOP  # Does this need to be safe-guarded?
    #   MUX12(BOT) -> BOT  # Does this need to be safe-guarded?
    #   APP(TOP,x) -> TOP
    #   APP(BOT,x) -> BOT
    # We additionally implement rules merging MUX11 into any other mux.
    # Note that when porting this to C, the C code should recycle terms
    # to reduce malloc overhead. See collect() calls in this function.
    # TODO handle levels in rules.
    # TODO check for stack self-collision and convert to BOT.

    port, term = safe(main.in1)
    is_normal = False  # similar to HVM runtime's init
    stack = [(is_normal, port, term)]

    while stack:
        is_normal, port, term = stack.pop()
        assert not isinstance(term, MAIN)

        # Normalize subterms.
        if not is_normal:
            # TODO what if isinstance(term, MUX12)? fork?
            if isinstance(term, MUX21):
                assert port != "out", port
                stack.append((True, port, term))
                stack.append((False, "out", safe(term.out)[-1]))
            elif isinstance(term, APP):
                stack.append((True, port, term))
                stack.append((False,) + safe(term.lhs))
            elif isinstance(term, JOIN):
                stack.append((True, port, term))
                # Could the C runtime fork here?
                stack.append((False,) + safe(term.lhs))
                stack.append((False,) + safe(term.rhs))
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
                link(term.pop("out"), term.pop("lhs"))
                collect(term)
                continue

            # term = APP lhs rhs
            # lhs = LAM var body
            # ------------------ APP-LAM aka beta
            # term = body
            # var = rhs
            if isinstance(lhs, LAM):
                assert lhs_port == "out"
                body_port, body = safe(lhs.body)
                link(term.pop(port), lhs.pop("body"))
                if lhs.var is not None:
                    link(lhs.pop("var"), term.pop("rhs"))
                collect(term)
                stack.append((False, body_port, body))
                continue

            # term = APP lhs rhs
            # lhs = MUX21 lhs1 lhs2
            # ---------------------- APP-MUX21 aka APP-PAR
            # term = MUX21 app1 app2
            # rhs1, rhs2 = MUX12 rhs
            # app1 = APP lhs1 rhs1
            # app2 = APP lhs2 rhs2
            if isinstance(lhs, MUX21):
                assert lhs_port == "out"
                rhs1, rhs2 = MUX12()(term.pop("rhs"))
                app1 = APP()(lhs.pop("in1"), rhs1)
                app2 = APP()(lhs.pop("in2"), rhs2)
                MUX21()(app1, app2, term.pop("out"))
                collect(term)
                # FIXME Is this right? HVM doesn't push here.
                stack.append((False,) + safe(app2))
                stack.append((False,) + safe(app1))
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
            # ------------------ JOIN-TOP-left
            # term = TOP
            if isinstance(lhs, TOP):
                link(term.pop("out"), term.pop("lhs"))
                collect(term)
                continue

            # term = JOIN lhs rhs
            # rhs = TOP
            # ------------------ JOIN-TOP-right
            # term = TOP
            if isinstance(rhs, TOP):
                link(term.pop("out"), term.pop("rhs"))
                collect(term)
                continue

            # term = JOIN lhs rhs
            # lhs = BOT
            # ------------------ JOIN-BOT-left
            # term = rhs
            if isinstance(lhs, BOT):
                link(term.pop("out"), term.pop("rhs"))
                collect(term)
                continue

            # term = JOIN lhs rhs
            # rhs = BOT
            # ------------------ JOIN-BOT-right
            # term = lhs
            if isinstance(rhs, BOT):
                link(term.pop("out"), term.pop("lhs"))
                collect(term)
                continue

        if isinstance(term, MUX11):
            assert port == "out"
            x_port, x = safe(term.in1)

            # out = MUX11 x
            # x = MUX11 in1
            # --------------- MUX11-MUX11
            # out = MUX11 in1
            if isinstance(x, MUX11):
                link(x.pop("out"), x.pop("in1"))
                collect(x)
                stack.append((False, "out", term))
                continue

            # out = MUX11 x
            # x = MUX21 in1, in2
            # ------------------- MUX11-MUX21
            # out = MUX21 in1 in2
            if isinstance(x, MUX21):
                link(term.pop("out"), term.pop("in1"))
                collect(term)
                stack.append((False, x_port, x))
                continue

            # out1 = MUX11 x
            # out2, x = MUX12 in1
            # ---------------------- MUX11-MUX12
            # out2, out1 = MUX12 in1
            if isinstance(x, MUX12):
                link(term.pop("out"), term.pop("in1"))
                collect(term)
                stack.append((False, x_port, x))
                continue

            # out1 = MUX11 x
            # x, out2 = MUX12 in1
            # ---------------------- MUX11-MUX12
            # out1, out2 = MUX12 in1
            if isinstance(x, MUX12):
                link(term.pop("out"), term.pop("in1"))
                collect(term)
                stack.append((False, x_port, x))
                continue

        if isinstance(term, MUX12):
            assert port in ("out1", "out2")
            x_port, x = safe(term.in1)

            # out1, out2 = MUX12 x
            # x = JOIN lhs, rhs
            # ------------------------- MUX12-JOIN
            # out1 = JOIN lhs1 rhs1
            # out2 = JOIN lhs2 rhs2
            # lhs1, lhs2 = MUX12 lhs
            # rhs1, rhs2 = MUX12 rhs
            if isinstance(x, LAM):
                assert x_port == "out"
                lhs1, lhs2 = MUX12()(x.pop("lhs"))
                rhs1, rhs2 = MUX12()(x.pop("rhs"))
                JOIN()(lhs1, rhs1, term.pop("out1"))
                JOIN()(lhs2, rhs2, term.pop("out2"))
                collect(term)
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
                body1, body2 = MUX12()(x.pop("body"))
                var1, out1 = LAM()(None, body1, term.pop("out1"))
                var2, out2 = LAM()(None, body2, term.pop("out2"))
                MUX21()(var1, var2, x.pop("var"))
                collect(term)
                continue

            # out1, out2 = MUX12 x
            # x = MUX11 in1
            # ---------------------- MUX12-MUX11
            # out1, out2 = MUX12 in1
            if isinstance(x, MUX11):
                assert x_port == "out"
                link(x.pop("out"), x.pop("in1"))
                collect(x)
                stack.append((False, port, term))
                continue

            if isinstance(x, MUX21):
                assert x_port == "out"

                # out1, out2 = MUX12 x
                # x = MUX21 in1 in2
                # -------------------- MUX-MUX-elim aka DUP-PAR (equal)
                # out1 = in1
                # out2 = in2
                if term.level == x.level:
                    link(x.pop("in1"), term.pop("out1"))
                    link(x.pop("in2"), term.pop("out2"))
                    collect(term)
                    continue

                # out1, out2 = MUX12 x
                # x = MUX21 in1 in2
                # -------------------- MUX-MUX-intro aka DUP-PAR (different)
                # out1 = MUX21 x11 x12
                # out2 = MUX21 x21 x22
                # x11, x21 = MUX12 in1
                # x12, x22 = MUX12 in2
                x11, x21 = MUX12()(x.pop("in1"))
                x12, x22 = MUX12()(x.pop("in2"))
                MUX21()(x11, x12, term.pop("out1"))
                MUX21()(x21, x22, term.pop("out2"))
                collect(term)
                continue

        if isinstance(term, MUX21):
            assert port == "out"

            # out = MUX21 x in2
            # x = MUX11 in1
            # ------------------- MUX21-MUX11-left
            # out = MUX12 in1 in2
            in1_port, in1 = safe(term.in1)
            if isinstance(in1, MUX11):
                link(in1.pop("out"), in1.pop("in1"))
                collect(in1)
                stack.append((False, port, term))
                continue

            # out = MUX21 in1 x
            # x = MUX11 in2
            # ------------------- MUX21-MUX11-right
            # out = MUX12 in1 in2
            in2_port, in2 = safe(term.in2)
            if isinstance(in2, MUX11):
                link(in2.pop("out"), in2.pop("in1"))
                collect(in2)
                stack.append((False, port, term))
                continue


################################################################################
# Parsing : text -> graph


re_varname = re.compile("[a-z_][a-z0-9_]*$")
re_numeral = re.compile("[0-9]+$")
Env = Dict[str, Tuple[str, Term]]


def parse(text: str) -> MAIN:
    """
    Parse a Polish notation string to create a term graph.

    :param str text: A lambda term in polish notation.
    :returns: The root term of an interaction net graph.
    :rtype: Term
    """
    text = re.sub("#.*[\n\r]", " ", text)  # remove line comments
    tokens = text.strip().split()
    tokens.reverse()
    env: Env = {}
    port, term = _parse(tokens, env)
    assert not tokens, f"Extra input: {' '.join(tokens)}"

    main = MAIN()
    link(("in1", main), (port, term))
    if DEBUG:
        validate(main)
    return main


def _parse(tokens: List[str], env: Env) -> Tuple[str, Term]:
    token = tokens.pop()

    if token == "BOT":
        return "out", BOT()

    if token == "TOP":
        return "out", TOP()

    if token == "JOIN":
        lhs_ = _parse(tokens, env)
        rhs_ = _parse(tokens, env)
        return JOIN()(lhs_, rhs_)

    if token == "APP":
        lhs_ = _parse(tokens, env)
        rhs_ = _parse(tokens, env)
        return APP()(lhs_, rhs_)

    if token == "LAM":
        name = tokens.pop()
        assert re_varname.match(name), name
        lam = LAM()

        # Parse in an modified environment.
        old = env.get(name)
        env[name] = "var", lam
        link(("body", lam), _parse(tokens, env))
        var_port, var = env.pop(name)
        if old is not None:
            env[name] = old

        if var is lam:  # Variable was never used.
            assert var_port == "var", var_port
            lam.pop("var")
        else:  # Variable was used at least once.
            assert isinstance(var, MUX12), type(var).__name__
            assert var_port == "out1", var_port
            assert var.in1 is not None
            assert var.out1 is None, var.out1
            assert var.out2 is not None
            # Eagerly eliminate the final mux.
            link(var.pop("in1"), var.pop("out2"))
            collect(var)
        return "out", lam

    if token == "LET":  # syntactic sugar
        name = tokens.pop()
        assert re_varname.match(name), name
        defn_port, defn = _parse(tokens, env)

        # Parse in an modified environment.
        old = env.get(name)
        env[name] = defn_port, defn
        result = _parse(tokens, env)
        var_port, var = env.pop(name)
        if old is not None:
            env[name] = old

        if var is defn:  # Definition was never used.
            collect(defn)
        else:  # Definition was used at least once.
            assert isinstance(var, MUX12), type(var).__name__
            assert var_port == "out1", var_port
            assert var.in1 is not None
            assert var.out1 is None, var.out1
            # Eagerly eliminate the final mux.
            if var.out2 is None:
                assert result == ("out2", var)
                result = safe(var.pop("in1"))
            else:
                link(var.pop("in1"), var.pop("out2"))
            collect(var)
        return result

    if token == "LETREC":  # syntactic sugar
        name = tokens.pop()
        assert re_varname.match(name), name
        mux = MUX12()

        # Parse in an modified environment.
        old = env.get(name)
        env[name] = "out1", mux
        body_port, body = _parse(tokens, env)
        var_port, var = env.pop(name)
        if old is not None:
            env[name] = old

        assert isinstance(var, MUX12)
        if var is mux:  # Recursion was never used.
            collect(mux)
            return body_port, body
        else:  # Recursion was used at least once.
            link((body_port, body), ("in1", mux))
            return "out2", mux

    if re_varname.match(token):
        name = token
        assert name in env, name
        # Add a MUX12 for each occurrence; the final MUX12 will later be removed.
        out1, out2 = MUX12()(env[name])
        env[name] = out1  # save for later
        return out2  # use now

    if re_numeral.match(token):
        # Create a Church numeral.
        n = int(token)
        term = LAM()
        body = LAM()
        link(("body", term), ("out", body))
        if n == 0:
            link(("body", body), ("var", body))
            return "out", term
        f: Tuple[str, Term] = "var", term
        x: Tuple[str, Term] = "var", body
        for _ in range(n - 1):
            f, f_temp = MUX12()(f)
            x = APP()(f_temp, x)
        APP()(f, x, ("body", body))
        return "out", term

    raise ValueError(f"Unhandled token: {token}")


if DEBUG >= 2:
    _parse_base = _parse

    def _parse(tokens: List[str], env: Env) -> Tuple[str, Term]:
        port, term = _parse_base(tokens, env)
        print(port, type(term).__name__)
        return port, term


################################################################################
# Readback : graph -> text


def readback(main: MAIN) -> str:
    env: Dict[Term, str] = {}
    port, term = safe(main.in1)
    tokens = _readback(port, term, env)
    tokens.reverse()
    return " ".join(tokens)


def _readback(port: str, term: Term, env: Dict[Term, str]) -> List[str]:
    assert not isinstance(term, MAIN)

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
