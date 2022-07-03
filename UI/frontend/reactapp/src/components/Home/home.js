import React, { useState } from "react";
import ProgressBar from 'react-bootstrap/ProgressBar';
import 'bootstrap/dist/css/bootstrap.min.css';
import axios from 'axios';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { Link } from "react-router-dom";


export default function Home() {
  const [files, setFiles] = useState('');
  const [uploadPercentage, setUploadPercentage] = useState(0)

  const notify = () => toast("Video uploaded successfully!");
  const notifyerror = () => toast.error('Error.. wrong file!', {
      position: "top-right",
      autoClose: 5000,
      hideProgressBar: false,
      closeOnClick: true,
      pauseOnHover: true,
      draggable: true,
      progress: undefined,
      });
    
  const onChange = e =>{
      setFiles(e.target.files[0])
    }

  const onSubmit = async e =>{
    e.preventDefault();
    const formData = new FormData();
    formData.append("video",files)
    axios.post('http://127.0.0.1:8000/api/video/', formData,{
      onUploadProgress: (data) =>{
        setUploadPercentage(Math.round((data.loaded / data.total)*100))
      },
    }).then((res) => {
            notify();
            files.isUploading = false;
            console.log('Done')
        })
        .catch((err) => {
            // inform the user
            notifyerror();
            console.error(err)
            
        });
}

  return (
    <div>
       <form className="" onSubmit={onSubmit}>
        <div className="flex flex-col items-center px-12 mt-36 text-xl text-blue-400 font-bold">
            <h1 className="text-xl md:text-5xl lg:text-6xl mb-4 text-black">
               Hello there!
            </h1>
            <h1 className="text-2xl md:text-4xl lg:text-5xl mb-16">
                Upload your video diving right here
            </h1>
              <input className="border-2 px-2 py-2 mb-4" type="file" onChange={onChange} accept="video/mp4,video/x-m4v,video/*"/>
              </div>
           <div className="px-96">
             { uploadPercentage > 0 && <ProgressBar now={uploadPercentage} active label={`${uploadPercentage}%`} /> }
              <ToastContainer />
            </div> 
            <div className="flex flex-row justify-center items-center"> 
              <Link to="videos" className="px-20 py-2 mt-12 rounded-lg no-underline border-2 bg-green-400 hover:bg-green-500 cursor-pointer text-black font-bold text-2xl">Watch your score</Link>
              <input type='submit' className=" px-24 py-2 mt-12 rounded-lg border-2 bg-yellow-400 hover:bg-yellow-500 cursor-pointer text-black font-bold text-2xl" value='Upload' />
            </div>
      </form>
  </div>
  );
}