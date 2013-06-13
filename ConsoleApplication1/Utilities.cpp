#include "Utilities.h"
#include <algorithm>
#include <Windows.h>
#include "dirent.h"
#include "MiniAiff.h"

std::vector<std::string> GetFiles(std::string& dir_path)
{
	std::vector<std::string>  res;
	DIR *dir;
	struct dirent *ent;
	if ( ( dir = opendir(dir_path.c_str()) ) != NULL) 
	{
		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) 
			if (ent->d_type == DT_REG)
				res.push_back( dir_path+std::string(ent->d_name) );
		closedir (dir);
	} 
	else 
	{
	  /* could not open directory */
	  perror ("");
	}

	return res;
}

std::vector< std::shared_ptr< Tensor<float> > > ReadAiffFiles( std::vector< std::string > files)
{
	size_t CHUNK_SIZE = 8192;
	std::vector< std::shared_ptr< Tensor<float> > > res;
	for (size_t file_ind=0; file_ind<files.size(); file_ind++)
	{
		char* filepath = new char[files[file_ind].size() + 1];
		std::copy(files[file_ind].begin(), files[file_ind].end(), filepath);
		filepath[files[file_ind].size()] = '\0';
		long channels = mAiffGetNumberOfChannels(filepath);
		float **data = mAiffAllocateAudioBuffer(channels, CHUNK_SIZE);

		// Read from input file
		long num_entries = mAiffReadData(filepath, data, 0, CHUNK_SIZE, channels);

		std::vector<size_t> audio_dims; audio_dims.push_back(num_entries);
		std::shared_ptr< Tensor<float> > audio_features(new Tensor<float>(audio_dims));
		for (long i=0; i<num_entries; i++)
			(*audio_features)[i] = (*data)[i];
		mAiffDeallocateAudioBuffer(data, channels);
		delete[] filepath;
		res.push_back(audio_features);
	}
	return res;
}