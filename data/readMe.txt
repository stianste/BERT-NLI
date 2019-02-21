The Text_chunks directory contains two directories:europe_data and non_europe_data,
	inside each directory there are directories for each country,
		inside each country dir there are directories for each user,
			which contain chunks of 100 tokenized sentences written by that user.
			
We used the europe_data for training (and testing in 10-fold cross validation) and the non_europe data for testing.

Each country name is also the label for the NLI task, except for:
* US, UK, Australia, Ireland and New-Zealand which their label is English
* Spain and Mexico which their label is Spanish 
* Germany and Austria which their label is German

The full used countries list is:
	UK, US, Ireland, Australia, New-Zealand: English
	Germany, Austria: German
	Spain, Mexico: Spanish
	Netherlands 
	Poland 
	France 
	Sweden 
	Finland 
	Romania 
	Portugal
	Greece 
	Italy 
	Turkey 
	Norway 
	Czech 
	Croatia 
	Russia 
	Estonia 
	Bulgaria 
	Hungry
	Serbia
	Lithuania 
	Slovenia 

We used only 104 users from each country each time, so the classes will be more equal. These users are selected randomly.

We also used only the median number of chunks for each user to make sure certain users won't over dominant the dataset.
	For the europe_data we used at most 3 chunks per user, selected randomly.
	For the non_europe data we used at most 17 chunks per user, selected randomly.