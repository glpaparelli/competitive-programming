use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use std::io::Write;
use std::env;

struct Input {
    segments: Vec<(u32, u32)>,
    queries: Vec<(u32, u32, u32)>,
}

fn read_input() -> Input {
    // Read the first line with n and m
    let mut line = String::new();
    io::stdin()
        .read_line(&mut line)
        .expect("Failed to read line");
    let mut iter = line.split_whitespace();
    let n: usize = iter.next().unwrap().parse().expect("Failed to parse n");
    let m: usize = iter.next().unwrap().parse().expect("Failed to parse m");
    
    let mut segments: Vec<(u32, u32)> = Vec::new();
    let mut queries: Vec<(u32, u32, u32)> = Vec::new();

    // Read n lines, each line with two u32, the endpoints of the segment
    for _ in 0..n {
        line.clear();
        io::stdin().read_line(&mut line).expect("Failed to read line");

        // Parse two positive integers from the line
        let read_line: Vec<u32> = line
            .split_whitespace()
            .map(|s| s.trim().parse().expect("Failed to parse line"))
            .collect();

        let segment = (read_line[0], read_line[1]);
        segments.push(segment);
    }
    
    // Read m lines, each line with three u32, the pars (i,j,k) of the query
    line.clear();
    for _ in 0..m {
        io::stdin().read_line(&mut line).expect("Failed to read line");
        let read_line: Vec<u32> = line
            .split_whitespace()
            .map(|s| s.trim().parse().expect("Failed to parse line"))
            .collect();

        let query = (read_line[0], read_line[1], read_line[2]);
        queries.push(query);
    }

    Input { segments, queries }
}

fn read_input_from_file(file_path: String) -> Input {
    let file = File::open(file_path).expect("Failed to open file");
    let mut reader = BufReader::new(file);

    let mut line = String::new();
    reader.read_line(&mut line).expect("Failed to read line");

    let mut iter = line.split_whitespace();
    let n: usize = iter.next().unwrap().parse().expect("Failed to parse n");
    let m: usize = iter.next().unwrap().parse().expect("Failed to parse m");

    let mut segments: Vec<(u32, u32)> = Vec::new();
    let mut queries: Vec<(u32, u32, u32)> = Vec::new();

    for _ in 0..n {
        line.clear();
        reader.read_line(&mut line).expect("Failed to read line");

        let read_line: Vec<u32> = line
            .split_whitespace()
            .map(|s| s.trim().parse().expect("Failed to parse line"))
            .collect();

        let segment = (read_line[0], read_line[1]);
        segments.push(segment);
    }
    
    for _ in 0..m {
        line.clear();
        reader.read_line(&mut line).expect("Failed to read line");
        let read_line: Vec<u32> = line
            .split_whitespace()
            .map(|s| s.trim().parse().expect("Failed to parse line"))
            .collect();

        let query = (read_line[0], read_line[1], read_line[2]);
        queries.push(query);
    }

    Input { segments, queries }
}

/// Return 1 if there is a position that is contained by exactly k segments
fn is_there(segments: &Vec<(u32, u32)>, query: (u32, u32, u32)) -> bool {
    let n = segments.len();
    let mut axis: Vec<i32> = vec![0; n + 1];    

    // Define the array axis: 
    // axis[i] = number of "active" segments that starts in i
    for segment in segments {
        axis[segment.0 as usize] += 1; 
        axis[(segment.1 + 1) as usize] -= 1;
    }

    // Compute (in-place) the prefix sum of axis
    // axis[i] = numbers of "currently" "active" segments
    for i in 0..n {
        axis[i+1] += axis[i];
    }

    // Map the active segments with positions:
    // map[#active_segments] -> positions with that number of active segments
    let mut map = vec![Vec::<usize>::new(); n];
    for i in 0..n+1 {
        map[axis[i] as usize].push(i);
    }

    // Syntax sugar to match the names of the parameters of the query
    let i = query.0 as usize;
    let j = query.1 as usize; 
    let k = query.2 as usize; 
    // For each positions where there are exactly k segments 
    // we binary search the position p that is in [i,j]
    segment_binary_search(&map[k], i, j)
}

/// Returns true if it is found an element p in positions that is in [i,j]
fn segment_binary_search(positions: &Vec<usize>, i:usize, j:usize) -> bool {
    let mut start = 0;
    let mut end = positions.len();

    // No positions with k "active" segments
    if end == 0 {
        return false;
    }

    while start <= end {
        let mid = start + (end - start)/2;
        let position = positions[mid]; 

        // Our p is position, position is in exactly k segments
        if i <= position && position <= j {
            return true; 
        }

        if end - start == 1 {
            return false;
        }

        if i > position {
            start = mid;
        } else {
            end = mid; 
        }
    }
    false
}

fn produce_outputs() -> io::Result<()> {
    // To use create directory inputs and my_outputs
    // In inputs place the inputs file, call this from main and run
    for i in 0..8 {
        if let Ok(current_dir) = env::current_dir() {
            // Build the full path to the file in the current working directory
            let input_file_name = format!("/inputs/input{}.txt", i);
            let input_file_path = current_dir.to_string_lossy().to_string() + &input_file_name;

            let input = read_input_from_file(input_file_path);

            // Create output file
            let output_file_name = format!("/my_outputs/output{}.txt", i);
            let output_file_path = 
                current_dir.to_string_lossy().to_string() + &output_file_name;
            let mut output_file = File::create(output_file_path)?;

            // Fill output file
            for query in input.queries {
                if is_there(&input.segments, query) {
                    writeln!(output_file, "1")?;
                } else {
                    writeln!(output_file, "0")?;
                }
            }
        }
    }
    Ok(())
}

fn main() {  
    // let res = produce_outputs();  

    let input = read_input_();
    for query in input.queries {
        if is_there(&input.segments, query) {
            println!("1");
        } else {
            println!("0");
        }
    }
}