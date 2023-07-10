clc;
clear;

syms theta gamma;

v1 = cos(theta) * cos(gamma);
v2 = sin(theta) * cos(gamma);
v3 = sin(gamma);
v = [v1; v2; v3];

dvdtheta = diff(v, theta);
dvdgamma = diff(v, gamma);

disp("dvdtheta:"); pretty(simplify(dvdtheta));
disp("norm(dvdtheta):"); pretty(simplify(sqrt(dvdtheta.' * dvdtheta)));
disp("dvdgamma:"); pretty(simplify(dvdgamma));
disp("norm(dvdgamma):"); pretty(simplify(sqrt(dvdgamma.' * dvdgamma)));

pretty(simplify(cross(v, dvdtheta)));
pretty(v.' * dvdtheta);

pretty(simplify(cross(v, dvdgamma)));
pretty(simplify(v.' * dvdgamma));

pretty(simplify(dvdtheta.' * dvdgamma));

pretty(simplify(dvdgamma.' * cross(v, dvdtheta)));