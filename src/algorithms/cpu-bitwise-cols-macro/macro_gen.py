# COL_TYPE_SIZE = 16
# COL_TYPE_SIZE = 32
COL_TYPE_SIZE = 64

COL_TYPE = f'std::uint{COL_TYPE_SIZE}_t'

lt_var = 'lt'; ct_var = 'ct'; rt_var = 'rt'
lc_var = 'lc'; cc_var = 'cc'; rc_var = 'rc'
lb_var = 'lb'; cb_var = 'cb'; rb_var = 'rb'

def count_bits_f(expr):
    return '__builtin_popcountll(' + expr + ')'

def cast(expr):
    return f'static_cast<{COL_TYPE}>({expr})'

def num_bits(num):
    CONST_LEN = 0
    return cast('0b' + bin(num)[2:].zfill(CONST_LEN))

def _1_at(N):
    return num_bits(1 << N)

def if_else(cond, if_expr, else_expr):
    return f'(({cond}) ? ({if_expr}) : ({else_expr}))'
def _and(expr1, expr2):
    return f'(({expr1}) & ({expr2}))'

class INNER_BITS:
    @staticmethod
    def site_mask(N):
        return num_bits((7 << (N - 1)))

    @staticmethod
    def center_mask(N):
        return num_bits((5 << (N - 1)))

    @staticmethod
    def offset_nei(offset, N, expr):
        if N > 6:
            return f'(({expr}) >> ({offset}))'
        else:
            return f'(({expr}) << ({offset}))'

    @staticmethod
    def count_nei(N, rc, cc, lc):
        rc_off = INNER_BITS.offset_nei(6, N, rc)
        cc_off = INNER_BITS.offset_nei(3, N, cc)
        lc_off = lc

        return count_bits_f(f'({rc_off} | {cc_off} | {lc_off})')

        # return f'({count_bits_f(rc)} + {count_bits_f(cc)} + {count_bits_f(lc)})'

    @staticmethod
    def inner_bit(N, rc, cc, lc):
        c_mask = INNER_BITS.center_mask(N)
        s_mask = INNER_BITS.site_mask(N)
        cell_mask = _1_at(N)

        alive_nei = INNER_BITS.count_nei(N, _and(rc, s_mask), _and(cc, c_mask), _and(lc, s_mask))
        cell_state_is_alive = _and(cc, cell_mask)

        alive_cell = cell_mask
        dead_cell = num_bits(0)

        return if_else(f'({cell_state_is_alive})',
                        if_else(f'(({alive_nei}) == 2 || ({alive_nei}) == 3)', alive_cell, dead_cell),
                        if_else(f'(({alive_nei}) == 3)', alive_cell, dead_cell))


    @staticmethod
    def compute_all_inner_bits(BITS, rc, cc, lc):
        all_exprs = []

        for N in range(1, BITS - 1):
            expr = INNER_BITS.inner_bit(N, rc, cc, lc)
            all_exprs.append(expr)

        return '(' + '|'.join([f'({e})' for e in all_exprs]) + ')'

class BOTTOM_CONSTS:
    SITE_MASK = num_bits(3 << (COL_TYPE_SIZE - 2))
    CENTER_MASK = num_bits(1 << (COL_TYPE_SIZE - 2))
    UP_BOTTOM_MASK = num_bits(1)
    CELL_MASK = num_bits(1 << (COL_TYPE_SIZE - 1))

    @staticmethod
    def offset_center_cols(expr, N):
        return f'({expr} >> {N})'

    @staticmethod
    def offset_top_bottom_cols(expr, N):
        return f'({expr} << {N})'


class TOP_CONSTS:
    SITE_MASK = num_bits(3)
    CENTER_MASK = num_bits(2)
    UP_BOTTOM_MASK = _1_at(COL_TYPE_SIZE - 1)
    CELL_MASK = num_bits(1)

    @staticmethod
    def offset_center_cols(expr, N):
        return f'({expr} << {N})'

    @staticmethod
    def offset_top_bottom_cols(expr, N):
        return f'({expr} >> {N})'


class END_BITS:
    @staticmethod
    def compute_side_col(consts, lc, cc, rc, _l, _c, _r):
        site_mask = consts.SITE_MASK
        center_mask = consts.CENTER_MASK
        up_bottom_mask = consts.UP_BOTTOM_MASK
        cell_mask = consts.CELL_MASK

        neighborhood = f'(' \
                       f'({consts.offset_center_cols(f"({lc} & {site_mask})", 7)}) | ' \
                       f'({consts.offset_center_cols(f"({cc} & {center_mask})", 5)}) | ' \
                       f'({consts.offset_center_cols(f"({rc} & {site_mask})", 3)}) | ' \
                       f'({consts.offset_top_bottom_cols(f"({_l} & {up_bottom_mask})", 2)}) | ' \
                       f'({consts.offset_top_bottom_cols(f"({_c} & {up_bottom_mask})", 1)}) | ' \
                       f'({_r} & {up_bottom_mask}))'

        alive_neighbours = count_bits_f(neighborhood)
        cell = _and(cc, cell_mask) 

        return if_else(
            f'({cell} != 0)',
            if_else(f'({alive_neighbours} < 2 || {alive_neighbours} > 3)', '0', cell_mask),
            if_else(f'({alive_neighbours} == 3)', cell_mask, '0')
        )

def entire_expr(
    lt, ct, rt,
    lc, cc, rc,
    lb, cb, rb):

    top = END_BITS.compute_side_col(TOP_CONSTS, lc, cc, rc, lt, ct, rt)
    bottom = END_BITS.compute_side_col(BOTTOM_CONSTS, lc, cc, rc, lb, cb, rb)
    inner = INNER_BITS.compute_all_inner_bits(COL_TYPE_SIZE, rc, cc, lc)

    return f'(({top})|({bottom})|({inner}))'

def wrap_to_macro(
    lt, ct, rt,
    lc, cc, rc,
    lb, cb, rb,
    expr):
    
    name = f'__{COL_TYPE_SIZE}_BITS__GOL_BITWISE_COL_COMPUTE'

    return f'#define {name}({lt}, {ct}, {rt}, {lc}, {cc}, {rc}, {lb}, {cb}, {rb}) ({expr})'

res = entire_expr(
    lt_var, ct_var, rt_var, 
    lc_var, cc_var, rc_var, 
    lb_var, cb_var, rb_var)

macro = wrap_to_macro(
    lt_var, ct_var, rt_var, 
    lc_var, cc_var, rc_var, 
    lb_var, cb_var, rb_var, res)

print(macro)