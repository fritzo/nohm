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
        self.level = 0

    @classmethod
    @property
    def principle_port(cls):
        return cls.ports[0]

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


# Consider refactoring (FAN,LEV) into oriented (MUX11,MUX12,MUX21).
class FAN(Term):  # aka delta
    """Binary sharing."""

    ports = "out", "in1", "in2"  # Similar to HVM's PAR,DP0,DP1
    out: Port
    in1: Port
    in2: Port


class LEV(Term):  # Asperti's BOHM triangle
    """Level stepping."""

    ports = "out", "in1"
    out: Port
    in1: Port


class LAM(Term):  # aka lambda aka abs
    """Lambda abstraction."""

    ports = "out", "var", "body"
    out: Port
    var: Port
    body: Port


class APP(Term):  # aka apply
    """Function application."""

    ports = "lhs", "rhs", "out"
    lhs: Port
    rhs: Port
    out: Port


def unlink(term: Term, port: str) -> None:
    assert port in term.ports
    setattr(term, port, None)


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
    if isinstance(term, FAN) and port != "out":
        # FIXME this needs to check for safe fans. See Asperti98a.
        # in1, _ = FAN out
        # ----------------
        # in1 = LEV out
        lev = LEV()
        lev.out = term.pop("out")
        if port == "in1":
            lev.in1 = term.pop("in2" if port == "in1" else "in1")
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
            if isinstance(term, FAN):
                assert port != "out", "unexpected sharing"
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
            lhs_port, lhs = safe(term.lhs)

            # term = APP lhs rhs
            # lhs = LAM var body
            # ------------------ APP-LAM aka beta
            # term = body
            # var = rhs
            if isinstance(lhs, LAM):
                assert lhs_port == "out"
                head_port, head = safe(term.pop(port))
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
            # lhs = FAN in1 in2
            # -------------------- APP-FAN aka APP-PAR
            # term = FAN app1 app2
            # rhs1, rhs2 = FAN rhs
            # app1 = APP in1 rhs1
            # app2 = APP in2 rhs2
            if isinstance(lhs, FAN) and lhs_port == "out":
                head_port, head = getattr(term, port)
                rhs_port, rhs = safe(term.pop("rhs"))
                in1_port, in1 = safe(lhs.pop("in1"))
                in2_port, in2 = safe(lhs.pop("in2"))
                collect(port, term)
                term = FAN()
                rhs_fan = FAN()
                app1 = APP()
                app2 = APP()
                link(rhs_fan, "out", rhs_port, rhs)
                link(app1, "lhs", in1_port, in1)
                link(app1, "rhs", "in1", rhs_fan)
                link(app2, "lhs", in2_port, in2)
                link(app2, "rhs", "in2", rhs_fan)
                link(term, "in1", "out", app1)
                link(term, "in2", "out", app2)
                link(head, head_port, "out", term)
                # FIXME Is this right? HVM doesn't push here.
                stack.append((False, "out", app2))
                stack.append((False, "out", app1))
                continue

        if isinstance(term, FAN):
            if port == "out":
                out_port, out = safe(term.out)

                if isinstance(out, FAN):
                    assert out_port == "out"
                    in1_port, in1 = safe(term.pop("in1"))
                    in2_port, in2 = safe(term.pop("in2"))
                    in3_port, in3 = safe(out.pop("in1"))
                    in4_port, in4 = safe(out.pop("in2"))

                    # in1, in2 = out
                    # in3, in4 = out
                    # --------------- FAN-FAN-elim aka DUP-PAR (equal)
                    # in1 = in3
                    # in2 = in4
                    if term.level == out.level:
                        link(in1, in1_port, in3_port, in3)
                        link(in2, in2_port, in4_port, in4)

                    # in1, in2 = FAN out
                    # in3, in4 = FAN out
                    # ------------------ FAN-FAN-intro aka DUP-PAR (different)
                    # x13, x14 = FAN in1
                    # x23, x23 = FAN in2
                    # x13, x23 = FAN in3
                    # x14, x24 = FAN in4
                    else:
                        fan1 = FAN()
                        fan2 = FAN()
                        fan3 = FAN()
                        fan4 = FAN()
                        link(in1, in1_port, "out", fan1)
                        link(in2, in2_port, "out", fan2)
                        link(in3, in3_port, "out", fan3)
                        link(in4, in4_port, "out", fan4)
                        link(fan1, "in1", "in1", fan3)
                        link(fan1, "in2", "in1", fan4)
                        link(fan2, "in1", "in2", fan3)
                        link(fan2, "in2", "in2", fan4)

                    collect(port, term)
                    collect(out_port, out)
                    continue

                # in1, in2 = FAN out
                # out = LAM var body
                # ----------------------- FAN-LAM aka DUP-LAM
                # in1 = LAM var1 body1
                # in2 = LAM var2 body2
                # var1, var2 = FAN var1
                # body1, body2 = FAN body
                if isinstance(out, LAM):
                    assert out_port == "out"
                    in1_port, in1 = safe(term.pop("in1"))
                    in2_port, in2 = safe(term.pop("in2"))
                    var_port, var = safe(out.pop("var"))
                    body_port, body = safe(out.pop("body"))
                    collect(port, term)
                    collect(out_port, out)
                    var_fan = FAN()
                    body_fan = FAN()
                    lam1 = LAM()
                    lam2 = LAM()
                    link(var_fan, "out", var_port, var)
                    link(body_fan, "out", body_port, body)
                    link(lam1, "var", "in1", var_fan)
                    link(lam1, "body", "in1", body_fan)
                    link(lam2, "var", "in2", var_fan)
                    link(lam2, "body", "in2", body_fan)
                    link(in1, in1_port, "out", lam1)
                    link(in2, in2_port, "out", lam2)
                    continue

    return safe(getattr(root, port))


################################################################################
# Parsing : text -> graph


re_varname = re.compile("[a-z_][a-z0-9_]*$")
re_int = re.compile("[0-9]+$")
Env = Dict[str, Tuple[str, Term]]


def parse(text: str) -> Term:
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
    unlink(term, port)
    if DEBUG:
        validate(term)
    assert not tokens, f"Extra input: {' '.join(tokens)}"
    return term


def _parse(tokens: List[str], env: Env) -> Tuple[str, Term]:
    token = tokens.pop()

    if re_varname.match(token):
        name = token
        assert name in env
        # Add a FAN node for each occurrence.
        # The final FAN is removed in _link_usage().
        var = FAN()
        user_port, user = env[name]
        link(var, "out", user_port, user)
        unlink(var, "in2")
        env[name] = "out", var
        return "in1", var

    if token == "APP":
        lhs_port, lhs = _parse(tokens, env)
        rhs_port, rhs = _parse(tokens, env)
        app = APP()
        link(app, "lhs", lhs_port, lhs)
        link(app, "rhs", rhs_port, rhs)
        return "out", app

    if token == "LAM":
        name = tokens.pop()
        assert re_varname.match(name)
        lam = LAM()
        env = env.copy()
        env[name] = "var", lam
        body_port, body = _parse(tokens, env)
        link(lam, "body", body_port, body)
        used_port, used = env[name]
        if used is lam:
            # Variable was never used.
            assert used_port == "var"
            unlink(lam, "var")
        else:
            # Variable was used at least once.
            assert used_port == "out"
            assert isinstance(used, FAN)
            assert used.in2 is None
            # Eagerly eliminate the final FAN-_ pair.
            user_port, user = safe(used.in1)
            link(lam, "var", user_port, user)
            collect("out", used)
        return "out", lam

    # LET is syntactic sugar on top of the term language.
    if token == "LET":
        name = tokens.pop()
        defn_port, defn = _parse(tokens, env)
        env = env.copy()
        env[name] = defn_port, defn
        body_port, body = _parse(tokens, env)
        used_port, used = env[name]
        if used is defn:
            # defn was never used.
            collect(defn_port, defn)
        else:
            # Variable was used at least once.
            assert used_port == "out"
            assert isinstance(used, FAN)
            assert used.in2 is None
            # Eagerly eliminate the final FAN-_ pair.
            user_port, user = safe(used.in1)
            used_port, used = safe(used.out)
            assert isinstance(used, FAN)
            link(used, used_port, user_port, user)
            used.pop("in1")
            used.pop("out")
            collect("in2", used)
        return body_port, body

    # LETREC is syntactic sugar on top of the term language.
    if token == "LETREC":
        name = tokens.pop()
        fan = FAN()
        env = env.copy()
        env[name] = "in1", fan
        body_port, body = _parse(tokens, env)
        used_port, used = env[name]
        assert isinstance(used, FAN)
        if used is fan:
            # Recursion was never used.
            collect("out", fan)
            return body_port, body
        else:
            # Recursion was used at least once.
            link(body, body_port, "out", fan)
            return "in2", fan

    # Create a Church numeral.
    if re_int.match(token):
        n = int(token)
        term = LAM()
        body = LAM()
        unlink(term, "out")
        link(term, "body", "out", body)
        if n == 0:
            unlink(term, "var")
            link(body, "body", "var", body)
            return "out", term
        f_port = "var"
        f: Term = term
        x_port = "var"
        x: Term = body
        for _ in range(n - 1):
            app = APP()
            fan = FAN()
            link(fan, "out", f_port, f)
            link(app, "lhs", "in1", fan)
            unlink(fan, "in2")
            f_port, f = "in2", fan
            x_port, x = "out", app
        app = APP()
        link(app, "out", "body", body)
        link(app, "lhs", f_port, f)
        link(app, "rhs", x_port, x)
        return "out", term

    raise ValueError(f"Unhandled token: {token}")


################################################################################
# Readback : graph -> text


def readback(term: Term) -> str:
    env: Dict[Term, str] = {}
    if not isinstance(term, (LAM, APP)):
        raise NotImplementedError(f"Unsupported term: {type(term).__name__}")
    port = "out"
    tokens = _readback(port, term, env)
    tokens.reverse()
    return " ".join(tokens)


def _readback(port: str, term: Term, env: Dict[Term, str]) -> List[str]:

    while isinstance(term, FAN) and port in ("in1", "in2"):
        port, term = getattr(term, "out")
        # TODO support sharing via LET
        # TODO support looping via LETREC

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

    if isinstance(term, APP):
        lhs = _readback(*safe(term.lhs), env)
        rhs = _readback(*safe(term.rhs), env)
        return rhs + lhs + ["APP"]

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
    terms = set()
    pending = {root}
    while pending:
        source = pending.pop()
        for source_port in source.ports:
            assert hasattr(source, source_port)
            d = getattr(source, source_port)
            if d is None:
                if isinstance(d, LAM) and source_port in ("out", "var"):
                    continue  # lambdas can have unused variables
                elif d is root and source_port == "out":
                    continue  # the root node can have an empty out port
                raise ValueError(f"Unlinked {type(source).__name__}.{source_port}")
            destin_port, destin = d
            if destin not in terms:
                terms.add(destin)
                pending.add(destin)
                assert hasattr(destin, destin_port)
                assert getattr(destin, destin_port) == (source_port, source)


__all__ = [
    "parse",
    "readback",
    "reduce",
    "validate",
]
