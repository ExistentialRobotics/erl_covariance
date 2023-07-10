clc;
clear;
syms x1 x2 y1 y2;
syms a;
x = [x1; x2];
y = [y1; y2];

k = exp(-a * (x - y).' * (x - y));
dkdx1 = diff(k, x1);
dkdx2 = diff(k, x2);
dkdx = [dkdx1; dkdx2];
pretty(simplify(dkdx));

dkdy1 = diff(k, y1);
dkdy2 = diff(k, y2);
dkdy = [dkdy1; dkdy2];
pretty(simplify(dkdy));

d2kdx1dy1 = diff(dkdx1, y1);
d2kdx1dy2 = diff(dkdx1, y2);
d2kdx2dy1 = diff(dkdx2, y1);
d2kdx2dy2 = diff(dkdx2, y2);
d2kdxdy = [...
    d2kdx1dy1, d2kdx1dy2; ...
    d2kdx2dy1, d2kdx2dy2  ...
];
pretty(simplify(d2kdxdy));
