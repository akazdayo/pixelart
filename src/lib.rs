use std::{collections::HashMap};
use std::slice;


pub struct Palette{
    color_name: Vec<u8>,
    color_key: Vec<Vec<u8>>,
    color_val: Vec<Vec<u8>>
}


#[no_mangle]
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[no_mangle]
pub fn color_change(r: u8, g: u8, b: u8,color_palette:Vec<Vec<u8>>, color_key:Vec<Vec<u8>>, color_val:Vec<Vec<u8>>) -> Palette {
    let key = vec![r, g, b];
    if color_key.contains(&key) {
        let value = color_key.iter().position(|x| x == &key).unwrap();
        //let value = &color_dict[&key];
        
        return Palette {
            color_name: color_val[value].clone(),
            color_key: color_key,
            color_val: color_val,
        };
    }

    let mut min_distance = f32::INFINITY;
    let mut color_name: Vec<u8> = vec![];

    for color in color_palette{
        let distance = ((color[0] - r).pow(2) + (color[1] - g).pow(2) + (color[2] - b).pow(2)) as f32;
        if distance < min_distance {
            min_distance = distance;
            color_name = color;
        }
    }

    let mut new_color_key = color_key.clone();
    let mut new_color_val = color_val.clone();

    new_color_key.push(vec![r,g,b]);
    new_color_val.push(color_name.clone());

    let value = Palette {
        color_name: color_name,
        color_key: new_color_key,
        color_val: new_color_val,
    };

    println!("{:?}", value.color_name);

    return value;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn it_works2() {
        let color_palette = vec![vec![0, 0, 0], vec![255, 255, 255]];
        let color_key = vec![vec![0, 0, 0], vec![255, 255, 255]];
        let color_val = vec![vec![0, 0, 0], vec![255, 255, 255]];
        let result = color_change(0, 0, 0, color_palette, color_key, color_val);
        assert_eq!(result.color_name, vec![0, 0, 0]);
    }

    #[test]
    fn it_works3(){
        let color_palette = vec![vec![0, 0, 0], vec![255, 255, 255]];
        let color_key = vec![vec![0, 0, 0], vec![255, 255, 255]];
        let color_val = vec![vec![0, 0, 0], vec![255, 255, 255]];
        let result = color_change(128, 128, 128, color_palette, color_key, color_val);
        assert_eq!(result.color_name, vec![255, 255, 255]);
    }
}
