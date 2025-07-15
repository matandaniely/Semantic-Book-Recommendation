from difflib import get_close_matches

def text_to_string(row):
    """ This function converts a row of book data into a string representation. """

    textual_representation = f"""Title: {row['title']}
    Author: {row['authors']}
    Categories: {row['categories']}
    Description: {row['description']}
    Publishing Year: {row['published_year']}
    Rating: {row['average_rating']}
    Number of Pages: {row['num_pages']}
    """
    return textual_representation


def find_title_in_question(question, titles, threshold=0.85):
    question_lower = question.strip().lower()
    matches = get_close_matches(question_lower, [t.lower() for t in titles], n=1, cutoff=threshold)
    return matches[0] if matches else None
