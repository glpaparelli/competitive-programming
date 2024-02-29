use std::cmp;
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use std::io::Write;
use std::env;

struct Input {
    cities: usize,
    days: usize,
    itineraries: Vec<Vec<usize>>,
}

fn read_input() -> Input {
    // Read the first line: cities and days
    let mut line = String::new();
    io::stdin()
        .read_line(&mut line)
        .expect("Failed to read line");
    let mut iter = line.split_whitespace();
    let cities: usize = iter.next().unwrap().parse().expect("Failed to parse cities");
    let days: usize = iter.next().unwrap().parse().expect("Failed to parse days");

    // Read #cities itineraries
    let mut itineraries: Vec<Vec<usize>> = Vec::new();

    for _ in 0..cities {
        line.clear();

        io::stdin()
            .read_line(&mut line)
            .expect("error")
        ;
        
        let itinerary: Vec<usize> = line
            .split_whitespace()
            .map(|x| x.parse().expect("error"))
            .collect()
        ; 

        assert_eq!(itinerary.len(), days);


        itineraries.push(itinerary);
    }
    Input { cities, days, itineraries }
}

fn read_input_from_file(file_path: String) -> Input {
    let file = File::open(file_path).expect("error");
    let mut reader = BufReader::new(file);

    let mut line = String::new();
    reader.read_line(&mut line).expect("Failed to read line");

    let mut iter = line.split_whitespace();
    let cities: usize = iter.next().unwrap().parse().expect("Failed to parse n");
    let days: usize = iter.next().unwrap().parse().expect("Failed to parse m");

    let mut itineraries: Vec<Vec<usize>> = Vec::new();

    for _ in 0..cities {
        line.clear();

        reader
            .read_line(&mut line)
            .expect("error")
        ;
        
        let itinerary: Vec<usize> = line
            .split_whitespace()
            .map(|x| x.parse().expect("error"))
            .collect()
        ;

        assert_eq!(itinerary.len(), days);

        itineraries.push(itinerary);
    }
    Input { cities, days, itineraries }
}

fn produce_outputs() -> io::Result<()> {
    // To use create directory inputs and my_outputs
    // In inputs place the inputs file, call this from main and run
    for i in 0..5 {
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

            writeln!(output_file, "{}", holiday_planning(input.itineraries, input.cities, input.days))?;
        }
    }
    Ok(())
}

fn main() {
    // produce_outputs();
    let input: Input = read_input();
    println!("{}", holiday_planning(input.itineraries, input.cities, input.days))
}

fn holiday_planning(itineraries: Vec<Vec<usize>>, cities: usize, days:usize) -> usize {
    // Dynamic Programming: asf[i] =  max #attractions seen up to day i
    let mut asf = vec![0; days+1];

    // First itinerary: the #attractions seen up "day"
    // is the sum of attractions that can be seen in that city
    let first_itinerary = &itineraries[0];
    for day in 0..days {
        asf[day+1] = asf[day] + first_itinerary[day];
    }

    // for each itinerary (aka for each city)
    for itinerary in itineraries.iter().skip(1) {
        // asf_curr_city[i] = #attractions seen up to 
        // day i if we spend i days in the current city
        let mut asf_curr_city = vec![0; days + 1];
        for day in 0..days {
            asf_curr_city[day+1] = asf_curr_city[day] + itinerary[day];
        }
        
        // This is the dp array that we now compute considering the current city
        // as a place where we might spend some days, if it improves #attractions
        let mut curr_asf = vec![0; days + 1];

        // For each of the vacantion days
        for day in 1..days + 1 {    
            // days "before" the current day (prev_days is the number of days of vacation)
            // before the current day
            for prev_days in 0..day + 1 {
                // to compute the maximum number of attractions take the maximum between: 
                // 1) #attractions seen if we do not spend in the curr city
                // 2) #attractions seen if we spend "prev_days" in the curr city
                //    plus the #attractions seen if we spend the remaing days 
                //    up to the current day in the previous "optimal" city
                curr_asf[day] = std::cmp::max(
                    curr_asf[day], // 1)
                    asf_curr_city[prev_days] + asf[day - prev_days] // 2)
                );      
            }
        }
        asf = curr_asf; // update the dp array with the curr_asf
    }
    asf[days]
}