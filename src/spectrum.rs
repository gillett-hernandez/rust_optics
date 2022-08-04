pub fn spectrum_cauchy_from_abbe_num(nd: f32, vd: f32) -> (f32, f32) {
    if vd == 0.0 {
        (nd, 0.0)
    } else {
        const LC: f32 = 0.6563;
        const LF: f32 = 0.4861;
        const LD: f32 = 0.587561;
        const LC2: f32 = LC * LC;
        const LF2: f32 = LF * LF;
        const C: f32 = LC2 * LF2 / (LC2 - LF2);
        let b = (nd - 1.0) / vd * C;
        (nd - b / (LD * LD), b)
    }
}

pub fn spectrum_eta_from_abbe_num(nd: f32, vd: f32, lambda: f32) -> f32 {
    let (a, b) = spectrum_cauchy_from_abbe_num(nd, vd);
    a + b / (lambda * lambda)
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_eta_func() {
        let (nd, vd) = (1.4584, 67.82);
        let eta = spectrum_eta_from_abbe_num(nd, vd, 550.0);
        println!("{}", eta);
    }
}

// static const float spectrum_ungamma[] =
// {
//   0.000000,0.080560,0.110395,0.132737,0.151280,0.167429,0.181896,0.195098,0.207307,0.218708,0.229437,0.239595,0.249261,0.258497,0.267353,0.275870,0.284083,0.292020,0.299707,0.307164,0.314409,0.321460,0.328330,0.335031,0.341576,0.347973,0.354232,0.360361,0.366368,0.372258,0.378039,0.383716,0.389294,0.394777,0.400170,0.405478,0.410703,0.415850,0.420922,0.425921,0.430851,0.435714,0.440513,0.445250,0.449927,0.454547,0.459110,0.463620,0.468078,0.472486,0.476845,0.481157,0.485422,0.489643,0.493821,0.497957,0.502052,0.506108,0.510125,0.514104,0.518046,0.521953,0.525825,0.529664,0.533469,0.537242,0.540983,0.544693,0.548374,0.552025,0.555647,0.559241,0.562808,0.566348,0.569861,0.573349,0.576811,0.580248,0.583662,0.587051,0.590417,0.593760,0.597081,0.600380,0.603657,0.606913,0.610149,0.613363,0.616558,0.619733,0.622888,0.626025,0.629142,0.632242,0.635323,0.638386,0.641432,0.644460,0.647472,0.650467,0.653445,0.656407,0.659353,0.662284,0.665199,0.668099,0.670983,0.673853,0.676709,0.679549,0.682376,0.685189,0.687988,0.690773,0.693545,0.696304,0.699050,0.701783,0.704503,0.707210,0.709905,0.712588,0.715259,0.717918,0.720565,0.723201,0.725825,0.728438,0.731039,0.733630,0.736210,0.738778,0.741336,0.743884,0.746421,0.748948,0.751465,0.753971,0.756468,0.758955,0.761432,0.763899,0.766357,0.768805,0.771244,0.773674,0.776095,0.778507,0.780910,0.783303,0.785689,0.788065,0.790433,0.792793,0.795144,0.797487,0.799821,0.802148,0.804466,0.806776,0.809079,0.811373,0.813660,0.815939,0.818211,0.820475,0.822732,0.824981,0.827222,0.829457,0.831684,0.833905,0.836118,0.838324,0.840523,0.842715,0.844901,0.847079,0.849251,0.851417,0.853576,0.855728,0.857874,0.860013,0.862146,0.864273,0.866393,0.868507,0.870615,0.872717,0.874813,0.876903,0.878987,0.881065,0.883137,0.885203,0.887264,0.889318,0.891368,0.893411,0.895449,0.897481,0.899508,0.901529,0.903545,0.905556,0.907561,0.909561,0.911556,0.913545,0.915529,0.917509,0.919483,0.921451,0.923415,0.925374,0.927328,0.929277,0.931221,0.933160,0.935095,0.937025,0.938949,0.940870,0.942785,0.944696,0.946602,0.948503,0.950401,0.952293,0.954181,0.956064,0.957944,0.959818,0.961688,0.963554,0.965416,0.967273,0.969126,0.970975,0.972820,0.974660,0.976496,0.978328,0.980156,0.981980,0.983800,0.985616,0.987428,0.989235,0.991039,0.992839,0.994635,0.996427,0.998216,1.000000
// };

// static inline float spectrum_rgb_to_p(const float lambda, const float *rgb)
// {
//   // smits-like smooth metamer construction, basis function match cie rgb backwards.
//   float p = 0.0f;
//   // const float corr[] = {0.467044, 0.368873, 0.351969};
//   // float red = rgb[0]*corr[0], green = rgb[1]*corr[1], blue = rgb[2]*corr[2];
//   float red = rgb[0], green = rgb[1], blue = rgb[2];
//   float cyan = 0, yellow = 0, magenta = 0;
//   const float white = fminf(red, fminf(green, blue));
//   red -= white; green -= white; blue -= white;
//   const int bin = (int)(10.0f*(lambda - 380.0f)/(720.0 - 380.0));
//   float ww = spectrum_s_white[bin];
//   p += white * ww;
//   if(green > 0 && blue > 0)
//   {
//     cyan = fminf(green, blue);
//     green -= cyan; blue -= cyan;
//   }
//   else if(red > 0 && blue > 0)
//   {
//     magenta = fminf(red, blue);
//     red -= magenta; blue -= magenta;
//   }
//   else if(red > 0 && green > 0)
//   {
//     yellow = fminf(red, green);
//     red -= yellow; green -= yellow;
//   }

//   float cw = spectrum_s_cyan[bin];
//   float mw = spectrum_s_magenta[bin];
//   float yw = spectrum_s_yellow[bin];
//   p += cw*cyan;
//   p += mw*magenta;
//   p += yw*yellow;
//   float rw = spectrum_s_red[bin];
//   float gw = spectrum_s_green[bin];
//   float bw = spectrum_s_blue[bin];
//   p += red * rw;
//   p += green * gw;
//   p += blue * bw;
//   return p;
// }

// static inline float spectrum_tex_to_p(const float lambda, const unsigned char *rgb)
// {
//   float frgb[3] = {spectrum_ungamma[rgb[0]], spectrum_ungamma[rgb[1]], spectrum_ungamma[rgb[2]]};
//   return spectrum_rgb_to_p(lambda, frgb);
// }

// static inline void spectrum_p_to_xyz(const float lambda, const float p, float *xyz)
// {
//   const float corr[] = {2.291406, 2.395276, 2.650796}; // matched to convert to and fro
//   float b[3];
//   spectrum_xyz(lambda, b);
//   for(int k=0;k<3;k++) xyz[k] = b[k]*p*corr[k];
// }

// /* static inline void spectrum_p_to_cam(const float lambda, const float p, float *cam) */
// /* { */
// /*   float xyz[3]; */
// /*   spectrum_p_to_xyz(lambda, p, xyz); */
// /*   colorspace_xyz_to_cam(xyz, cam); */
// /* } */
// static inline void spectrum_p_to_rgb(const float lambda, const float p, float *rgb)
// {
//   float xyz[3];
//   spectrum_p_to_xyz(lambda, p, xyz);
//   spectrum_xyz_to_rgb(xyz, rgb);
// }

// // p = 1/(700-400)
// static inline float spectrum_sample_lambda(const float rand, float *pdf)
// {
//   if(pdf) *pdf = 1.0f/(700.0f - 300.0f);
//   return 400.0f + 300.0f*rand;
// }

// static inline float spectrum_lambda_pdf(const float lambda)
// {
//   return 1.0f/(700.0f - 400.0f);
// }

// #if 0
// // p ~ lum
// static inline float spectrum_sample_lambda(const float rand, float &p)
// {

// }

// static inline void spectrum_print_info(FILE *fd)
// {
//   //fprintf(fd, "spectrum : smits-style reconstruction\n");
// }
// #endif

// #endif
