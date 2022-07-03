import React, { useState, useEffect } from "react";
import axios from "axios";

let BASE_URL='http://localhost:8000';

function Videos() {
    const [videos, setVideos] = useState([]);
    useEffect(() => {
      const fetchBlogs = async () => {
        try {
          const response = await axios.get(`${BASE_URL}/api/video/`);
          setVideos(response.data);
        } catch (err) {}
        console.log('cant fetch data !');
      };
      fetchBlogs();
    }, []);

    return (
        <div id="videos" className='flex flex-col justify-center items-center pt-24 '>
            <h1> All Video Informations </h1>
    <table className="table-auto mt-12 w-full md:w-10/12 lg:w-10/12">
        <thead className="bg-yellow-400 rounded-t-lg">
          <tr>
              {/* <th className="p-3 text-sm font-semibold tracking-wide text-center">Video ID</th> */}
              <th className="p-3 text-sm font-semibold tracking-wide text-center">Video Link</th>
              <th className="p-3 text-sm font-semibold tracking-wide text-center">Score</th>
          </tr>
		    </thead>
        
        {/* {videos.map(post => {
                    return( */}
		    <tr className="bg-gray-100 border-b-2 hover:bg-gray-300">
              {/* <td className="text-center">{videos.id}</td> */}
              <td className="text-center">{videos.video}</td>
              <td className="text-justify">{videos.score}</td>
          </tr>
            {/* )

        })
        } */}
        </table>

        </div>
    );
}

export default Videos;