// #pragma once

use crate::math::*;

#[derive(Copy, Clone, Debug)]
pub enum LensType {
    Solid,
    Air,
    Aperture,
}

#[derive(Copy, Clone, Debug)]
pub struct LensElement {
    pub radius: f32,
    pub thickness_short: f32,
    pub thickness_mid: f32,
    pub thickness_long: f32,
    pub anamorphic: bool,
    pub lens_type: LensType,
    pub ior: f32, // index of refraction
    pub vno: f32, // abbe number
    pub housing_radius: f32,
    pub aspheric: i32,
    pub correction: f32x4,
}

impl LensElement {
    pub fn get_thickness(self, mut zoom: f32) -> f32 {
        if zoom < 0.5 {
            zoom *= 2.0;
            self.thickness_short * (1.0 - zoom) + self.thickness_mid * zoom
        } else {
            zoom -= 0.5;
            zoom *= 2.0;
            self.thickness_mid * (1.0 - zoom) + self.thickness_long * zoom
        }
    }
    pub fn get_aperture_radius(slice: &[Self]) -> f32 {
        for elem in slice {
            match elem.lens_type {
                LensType::Aperture => {
                    return elem.housing_radius;
                }
                _ => {
                    continue;
                }
            }
        }
        0.0
    }
    pub fn get_aperture_pos(slice: &[Self], zoom: f32) -> f32 {
        let mut pos = 0.0;
        for elem in slice {
            match elem.lens_type {
                LensType::Aperture => {
                    break;
                }
                _ => {
                    pos += elem.get_thickness(zoom);
                }
            }
        }
        pos
    }
}

// int lens_configuration(lens_element_t *l, const char *filename, int max)
// {
//   FILE *f = fopen(filename, "rb");
//   if(!f) return 0;
//   int cnt = 0;

//   float last_ior = 1.0f;
//   float last_vno = 0.0f;
//   float scale = 1.0f;
//   while(1)
//   {
//     lens_element_t lens;
//     memset(&lens, 0, sizeof(lens_element_t));
//     char line[2048];
//     if(fscanf(f, "%[^\n]", line) == EOF) break;
//     if(fgetc(f) == EOF) break; // new line

//     char *in = line;

//     if(!strncmp(line, "#!scale", 7))
//     {
//       scale = atof(line + 8);
//       continue;
//     }
//     // munch comment
//     if(!strncmp(line, "//", 2) || !strncmp(line, "#", 1)) continue;
//     while(in[0] == '\t' || in[0] == ' ') in++;
//     lens.radius = scale * strtof(in, &in);
//     if(lens.radius == 0.0f) break;
//     while(in[0] == '\t' || in[0] == ' ') in++;
//     lens.thickness_short = scale * strtof(in, &in);
//     while(in[0] == '\t' || in[0] == ' ') in++;
//     if(in[0] == '/')
//       lens.thickness_mid = scale * strtof(in+1, &in);
//     else
//       lens.thickness_mid = lens.thickness_short;
//     while(in[0] == '\t' || in[0] == ' ') in++;
//     if(in[0] == '/')
//       lens.thickness_long = scale * strtof(in+1, &in);
//     else
//       lens.thickness_long = lens.thickness_short;
//     if(lens.thickness_short == 0.0f) break;
//     if(lens.thickness_mid   == 0.0f) break;
//     if(lens.thickness_long  == 0.0f) break;

//     while(in[0] == '\t' || in[0] == ' ') in++;
//     if(!strncmp(in, "cx_", 3))
//     {
//       lens.anamorphic = 1;
//       in += 3;
//     }
//     int i=0;
//     while(in[0] != '\t' && in[0] != ' ' && in[0] != '\0') lens.material[i++] = in++[0];
//     lens.material[i] = '\0';
//     if(!strcasecmp(lens.material, "air"))
//     {
//       lens.ior = 1.0f;
//       lens.vno = 0.0f;
//     }
//     else if(!strcasecmp(lens.material, "iris"))
//     {
//       lens.ior = last_ior;
//       lens.vno = last_vno;
//     }
//     else
//     {
//       while(in[0] == '\t' || in[0] == ' ') in++;
//       lens.ior = strtof(in, &in);
//       while(in[0] == '\t' || in[0] == ' ') in++;
//       lens.vno = strtof(in, &in);
//     }
//     last_ior = lens.ior;
//     last_vno = lens.vno;
//     if(lens.ior == 0.0f) break;

//     while(in[0] == '\t' || in[0] == ' ') in++;
//     lens.housing_radius = scale * strtof(in, &in);
//     if(lens.housing_radius == 0.0f) break;

//     lens.aspheric = 0;
//     for(int i = 0; i < 4; i++)
//       lens.aspheric_correction_coefficients[i] = 0;

//     while(in[0] == '\t' || in[0] == ' ') in++;
//     if(!strncmp(in, "#!aspheric=", 11))
//     {
//       in += 11;
//       lens.aspheric = strtol(in, &in, 10);
//       // munch comma
//       in++;
//       for(int i = 0; i < 4; i++, in++)
//         lens.aspheric_correction_coefficients[i] = strtof(in, &in);
//     }

//     l[cnt++] = lens;

//     if(cnt >= max) break;
//   }
//   fclose(f);
//   return cnt;
// }

// static inline void lens_canonicalize_name(const char *filename, char *out)
// {
//   const char *start = filename;
//   const char *end = filename;
//   const char *c = filename;
//   for(;*c!=0;c++) if(*c == '/') start = c+1;
//   end = c;
//   int i=0;
//   for(;start != end;start++)
//   {
//     if(*start == '.') break;
//     else if(*start >= 65  && *start <= 90) // caps
//     {
//       if(i) out[i++] = ' ';
//       out[i++] = *start + 32;
//     }
//     else if(*start >= 48 && *start <= 59) // numbers
//       out[i++] = *start;
//     else if(*start < 97) // special
//       out[i++] = ' ';
//     else
//       out[i++] = *start;
//   }
//   out[i++] = 0;
// }
