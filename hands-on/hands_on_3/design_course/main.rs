use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use std::io::Write;
use std::env;

struct Input {
    topics: Vec<(u32, u32)>,
}

fn read_input() -> Input {
    let mut line = String::new();
    io::stdin().read_line(&mut line).expect("Failed to read line");
    let n: u32 = line.trim().parse().expect("Failed to parse n");
    
    let mut topics: Vec<(u32, u32)> = Vec::new();

    // Read n lines, each line with two u32, the endpoints of the segment
    for _ in 0..n {
        line.clear();
        io::stdin().read_line(&mut line).expect("Failed to read line");

        // Parse two positive integers from the line
        let read_line: Vec<u32> = line
            .split_whitespace()
            .map(|s| s.trim().parse().expect("Failed to read parse"))
            .collect();

        let course = (read_line[0], read_line[1]);
        topics.push(course);
    }

    Input {topics}
}

fn read_input_from_file(file_path: String) -> Input {
    let file = File::open(file_path).expect("Failed to read line");
    let mut reader = BufReader::new(file);
    
    let mut line = String::new();
    reader.read_line(&mut line).expect("Failed to read line");
    let n: u32 = line.trim().parse().expect("Failed to parse n");

    let mut topics: Vec<(u32, u32)> = Vec::new();

    for _ in 0..n {
        line.clear();
        
        reader.read_line(&mut line).expect("Failed to read line");
        // Parse two positive integers from the line
        let read_line: Vec<u32> = line
            .split_whitespace()
            .map(|s| s.trim().parse().expect("Failed to parse line"))
            .collect();

        let course = (read_line[0], read_line[1]);
        topics.push(course);
    }

    Input { topics }
}

fn produce_outputs() -> io::Result<()> {
    // To use create directory inputs and my_outputs
    // In inputs place the inputs file, call this from main and run
    for i in 0..11 {
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

            writeln!(output_file, "{}", design_a_course(input.topics))?;
        }
    }
    Ok(())
}

fn design_a_course(mut topics: Vec<(u32, u32)>) -> u32 {
    let n = topics.len();

    // Sort topics by beauty
    topics.sort_by(|a,b| a.0.cmp(&b.0));

    // lis[i] = LIS of the prefix [0..i-1] of the array
    let mut lis: Vec<u32> = vec![1; n];

    for i in 0..n {
        for j in 0..i {
            // 1) The current topic i is harder than the j-th topic
            //      - pedagodic requirement
            // 2) The current topic i is more beautiful that the j-th topic
            //      - students pickiness contraint
            // 3) The longest increasing subsequence of topics that ends in the topic i
            //    is smaller that the longest increasing subsequence that ends in the 
            //    topic j plus the current topic i
            //      - dp recurrence  
            if topics[i].1 > topics[j].1 && // 1)
               topics[i].0 > topics[j].0 && // 2)
               lis[i] < lis[j] + 1 // 3)
            {
                lis[i] = lis[j] + 1;
            }
        }
    }
    *lis.iter().max().unwrap()
}

fn main() {
    // produce_outputs();
    let input = read_input();
    println!("{}", design_a_course(input.topics));
}

