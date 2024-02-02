fn meetings_in_a_room(meetings: &mut Vec<(i32, i32)>) -> i32 {
    meetings.sort_by(|a, b| a.1.cmp(&b.1));
    let mut time_limit = meetings[0].1;
    let mut result = 1;

    for meeting in meetings.iter().skip(1) {
        // if the current meeting starts after the end of the last 
        // meeting than we can have the meeting in the room
        if meeting.0 > time_limit {
            result += 1;
            time_limit = meeting.1;
        }
    }
    result
}

fn main() {
    let mut meetings1: Vec<(i32, i32)> = vec![(1,2),(3,4),(0,6),(5,7),(8,9),(5,9)];
    let test1 = meetings_in_a_room(&mut meetings1);
    assert_eq!(test1, 4);

    let mut meetings2: Vec<(i32, i32)> = vec![(10,20),(12,25),(20,30)];
    let test2 = meetings_in_a_room(&mut meetings2);
    assert_eq!(test2, 1);
}
