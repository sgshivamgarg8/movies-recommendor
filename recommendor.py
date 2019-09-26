import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from surprise import Reader, Dataset, SVD#, evaluate
import warnings; warnings.simplefilter('ignore')

#inputUserId = int(input("Enter User Id - "))
#inputMovieName = input("Enter Movie Name - ")

print("Building Recommendation Model........................")

#Read Movies_metadata
md = pd.read_csv('Datasets/movies_metadata.csv')
md = md.drop([19730, 29503, 35587]) #These rows have bad formatted data
#Add genre and year column to metadata
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
#print(md.head())

'''
#Simple Rating Based Recommendor

#create a column of vote counts and vote_average per movie
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')

#store mean value of average votes in C
C = vote_averages.mean()
#print(C) #C = 5.244896612406511

#m = minimum votes required to be in top 250 list
m = vote_counts.quantile(0.95)
#print(m) #m = 434

#create a table of movies qualified with some columns in it like title, year, vote_counts, vote_averages
qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
#print(qualified.shape)

#function to calculate weighted rating of each movie using IMDb formula
def weighted_rating(x):
	v = x['vote_count']
	R = x['vote_average']
	return (v/(v+m) * R) + (m/(m+v) * C)

#apply fucntion on qualified table to add a column of 'wr' of each movie
qualified['wr'] = qualified.apply(weighted_rating, axis=1)	

#sort qualified table according to 'wr' high to low and get top 250 from that list
qualified = qualified.sort_values('wr', ascending=False).head(250)
#print(qualified)

#Break multiple genre of a movie into single and create multiple rows with all the genre for a movie
s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)
#print(gen_md.head())

#Function to get top 250 movies for a particular genre
def build_chart(genre, percentile=0.85):
	df = gen_md[gen_md['genre'] == genre]
	vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
	vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
	C = vote_averages.mean()
	m = vote_counts.quantile(percentile)
	
	qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
	qualified['vote_count'] = qualified['vote_count'].astype('int')
	qualified['vote_average'] = qualified['vote_average'].astype('int')
	
	qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
	qualified = qualified.sort_values('wr', ascending=False).head(250)
	
	return qualified

#print(build_chart('Crime'))
'''


############## Content-Based Recommendor ###################


#Read links.csv file contains ids
links_small = pd.read_csv('Datasets/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
#print(links_small)

md['id'] = md['id'].astype('int')
#print(md.shape)

#Filter out the movies whose id is present in links.csv file and store in smd
smd = md[md['id'].isin(links_small)]
#print(smd.shape)

'''
## Movie Description Based Recommendation

#Attach tagline in description column at the end
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

#Count the frequency of the words from the description and give them score
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
#print(tfidf_matrix)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#print(cosine_sim[0])

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

#print(get_recommendations("The Dark Knight"))
'''

### Metadata Based Recommendation

credits = pd.read_csv('Datasets/credits.csv')
keywords = pd.read_csv('Datasets/keywords.csv') 

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')

#Merge credits, keywords on md dataframe
md = md.merge(credits, on='id')
md = md.merge(keywords, on='id') 

smd = md[md['id'].isin(links_small)]

smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
#Add cast_size and crew_size column in smd
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

#Function to get Director name for each movie
def get_director(x):
	for i in x:
		if i['job'] == 'Director':
			return i['name']
	return np.nan

#Add Director column in smd
smd['director'] = smd['crew'].apply(get_director)

#Get Top 3 Actors for each movie
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

#Strip whitespaces from cast and convert to lowercase
smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

#Strip whitespaces from director and convert to lowercase and mention director 3 times to increase director's weightage relative to actors
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x, x, x])

#Create a series of keywords for a particular movie
s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
#print(s[:5])

#Remove keywords ocurring only once
s = s[s > 1]

stemmer = SnowballStemmer('english')
#print(stemmer.stem('dogs'))  # dog

def filter_keywords(x):
	words = []
	for i in x:
		if i in s:
			words.append(i)
	return words

#Filter and remove keywords having less than 2 frequency 
smd['keywords'] = smd['keywords'].apply(filter_keywords)

#make all plural keywords singular to neglect the difference and strip and lowercase
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

#Join keyword+cast+dir+genres
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

#Calculae cosine similarity 
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

''' #Comment out functions not needed in hybrid engine
def get_recommendations(title):
	idx = indices[title]
	sim_scores = list(enumerate(cosine_sim[idx]))
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	sim_scores = sim_scores[1:31]
	movie_indices = [i[0] for i in sim_scores]
	return titles.iloc[movie_indices]
#print(get_recommendations("The Dark Knight"))

def weighted_rating2(x):
	movies,m,C = get_top25_similar_movies()
	v = x['vote_count']
	R = x['vote_average']
	return (v/(v+m) * R) + (m/(m+v) * C)

# Function to improve the recommendation based on ratings

def get_top25_similar_movies():
	idx = indices[title]
	sim_scores = list(enumerate(cosine_sim[idx]))
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	sim_scores = sim_scores[1:26]
	movie_indices = [i[0] for i in sim_scores]
	movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average']]
	vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
	vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
	C = vote_averages.mean()
	m = vote_counts.quantile(0.60)
	return movies,m,C

def improved_recommendations(title):
	movies,m,C = get_top25_similar_movies()
	qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
	qualified['vote_count'] = qualified['vote_count'].astype('int')
	qualified['vote_average'] = qualified['vote_average'].astype('int')
	qualified['wr'] = qualified.apply(weighted_rating2, axis=1)
	qualified = qualified.sort_values('wr', ascending=False).head(10)
	return qualified
''' #Comment out functions not needed in hybrid engine

# title = "Pulp Fiction"
# print(improved_recommendations(title))


######### Collaborative Filtering

reader = Reader()
ratings = pd.read_csv('Datasets/ratings_small.csv')
#print(ratings.head())

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)
svd = SVD() #Singular Value Decomposition

#Evaluate
#evaluate(svd, data, measures=['RMSE', 'MAE']) #Mean RMSE = 0.8963, Mean MAE = 0.6896

#Train
trainset = data.build_full_trainset()
svd.train(trainset)

#print(ratings[ratings['userId'] == 1])

#print(svd.predict(1, 302, 3)) #userId=1, movieId=302

######################## Hybrid Engine ############################

#Function to convert tmdbId to int
def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

id_map = pd.read_csv('Datasets/links_small.csv')[['movieId', 'tmdbId']] #Read links_small.csv
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int) #convert tmdbId to int
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title') #Map tmdbId, id, title

indices_map = id_map.set_index('id')

#Hybrid function
def hybrid(userId, title):
    idx = indices[title]
    #Calculate similarity score 
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26] #Get top 25 similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'year', 'vote_count', 'vote_average', 'id']]
    #Apply Algorithm on movies to calculate estimated scores and create 'est' column
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False) #Sort movies acc to estimates
    return movies.head(10) #Return top 10 movies

result = hybrid(1,"Pulp Fiction")
result = result.reset_index()
result = result.drop(columns="index")
result = result.drop(columns="id")
print(result)
# print(hybrid(1,"Pulp Fiction"))
#print(hybrid(inputUserId, inputMovieName))