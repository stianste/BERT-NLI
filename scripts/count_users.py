import os

def get_users_from_dir(data_dir):
    usernames = set()
    for username in os.listdir(data_dir):
        usernames.add(username)

    return usernames

europe_users = get_users_from_dir(
    './data/RedditL2/text_chunks/europe_data/reddit.Lithuania.txt.tok.clean') 
non_europe_users = get_users_from_dir(
    './data/RedditL2/text_chunks/non_europe_data/reddit.Lithuania.txt.tok.clean') 

print('Num europe Lit:', len(europe_users))
print('Num non europe Lit:', len(non_europe_users))
print('Intersection:', len(europe_users.intersection(non_europe_users)))

europe_users = get_users_from_dir(
    './data/RedditL2/text_chunks/europe_data/reddit.Slovenia.txt.tok.clean') 
non_europe_users = get_users_from_dir(
    './data/RedditL2/text_chunks/non_europe_data/reddit.Slovenia.txt.tok.clean') 

print('Num europe Slovenia:', len(europe_users))
print('Num non europe Slovenia:', len(non_europe_users))
print('Intersection:', len(europe_users.intersection(non_europe_users)))
