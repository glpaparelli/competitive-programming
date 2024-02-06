use std::collections::BTreeMap;
use std::collections::HashMap;

#[derive(Copy, Clone, Debug)]
struct Frog {
    position: u32, 
    reach: u32, 
    index: u32, 
    eaten: u32
}

#[derive(Copy, Clone, Debug)]
struct Mosquito {
    position: u32, 
    value: u32, 
}

// using a BST or a sorted structure (the key point is to compute in O(1) 
// the predecessor) we obtain the right performances (instead now
// we have to iter through maybe_eaters and leftovers everytime)
// but I cant make it work in rust
fn frogs_and_mosquitoes(mut frogs: Vec<Frog>, mosquitoes: Vec<Mosquito>) {
    // position -> frog
    let mut axis: BTreeMap<u32, Frog> = BTreeMap::new();
    // landing position -> mosquitoes there
    let mut mosquito_to_eat: BTreeMap<u32, Mosquito> = BTreeMap::new();
    // store the mosquitoes that cannot be eaten
    let mut leftover_mosquitoes: Vec<Mosquito> = Vec::new();

    let mut res: HashMap<u32, Frog> = HashMap::new();

    // sort frogs by their position
    frogs.sort_by_key(|f: &Frog| f.position);

    // frogs do not overlap in the axis
    let mut current_reach = frogs[0].reach;
    axis.insert(frogs[0].position, frogs[0]);
    
    for frog in frogs.iter().skip(1) {
        if frog.reach > current_reach {
            axis.insert(std::cmp::max(current_reach+1, frog.position), *frog);
            current_reach = frog.reach
        } else {
            axis.insert(frog.position, *frog);
        }
    }
    
    let mut count;

    for mosquito in mosquitoes {
        count = 0;

        // insert the mosquito in the mosquito-to-eat-in-that-posiion
        // all the frogs that are before the landing position
        for f in axis.range(..(mosquito.position+1)) {
            let maybe_eaters = axis.range(..(mosquito.position+1)).count();

            let mut eating_frog = *f.1;
            // skip the frogs that cannot reach the frog
            if eating_frog.reach < mosquito.position {
                count += 1;
                // no frog could eat this mosquito
                if count == maybe_eaters {
                    leftover_mosquitoes.push(mosquito);
                }
                continue;
            }

            // eating_frog now eats the current mosquito
            eating_frog.eaten += 1; 
            eating_frog.reach += mosquito.value;
            mosquito_to_eat.remove(&mosquito.position);
            
            // this frog has now increased reach, maybe it can eat more
            // we need to check the old mosquitoes if they can be eaten
            let mut again = true;
            while again == true {
                again = false;

                for i in 0..leftover_mosquitoes.len() {
                    // the frogs eat again
                    if leftover_mosquitoes[i].position <= eating_frog.reach {
                        // println!("X: frog in pos {} has eaten the mosquito in pos {}", eating_frog.position, leftover_mosquitoes[i].position);
                        eating_frog.eaten += 1; 
                        eating_frog.reach += leftover_mosquitoes[i].value;
                        leftover_mosquitoes.remove(i);
                        again = true;
                    }
                }
            }

            res.insert(eating_frog.position, eating_frog);
        }
    }

    let mut res_array: Vec<Frog> = res.values().cloned().collect();
    res_array.sort_by_key(|f| f.index);

    for f in res_array {
        println!("{} {}", f.eaten, f.reach - f.position);
    }

}

fn main() {
    // test 1
    let mut f1: Frog = Frog{position: 10, reach: 12, index: 0, eaten: 0};
    let f2 = Frog{position: 15, reach: 15, index: 1, eaten: 0};
    let f3 = Frog{position: 6, reach: 7, index: 2, eaten: 0};
    let f4 = Frog{position: 0, reach: 1, index: 3, eaten: 0};

    let mut frogs: Vec<Frog> = Vec::new();
    frogs.insert(0, f1);
    frogs.insert(1, f2);
    frogs.insert(2, f3);
    frogs.insert(3, f4);

    let mut m1 = Mosquito{position: 110, value: 10};
    let mut m2 = Mosquito{position: 1, value: 1};
    let m3 = Mosquito{position: 6, value: 0};
    let m4 = Mosquito{position: 15, value: 10};
    let m5 = Mosquito{position: 14, value: 100};
    let m6 = Mosquito{position: 12, value: 2};

    let mut mosquitoes: Vec<Mosquito> = Vec::new();
    mosquitoes.push(m1);
    mosquitoes.push(m2);
    mosquitoes.push(m3);
    mosquitoes.push(m4);
    mosquitoes.push(m5);
    mosquitoes.push(m6);

    // frogs_and_mosquitoes(frogs, mosquitoes);

    // test2
    f1 = Frog{position: 10, reach: 12, index: 0, eaten: 0};
    let mut frogs1: Vec<Frog> = Vec::new();
    frogs1.insert(0, f1);

    m1 = Mosquito{position: 20, value: 2};
    m2 = Mosquito{position: 12, value: 1};
    let mut mosquitoes1: Vec<Mosquito> = Vec::new();
    mosquitoes1.push(m1);
    mosquitoes1.push(m2);

    frogs_and_mosquitoes(frogs1, mosquitoes1);
}