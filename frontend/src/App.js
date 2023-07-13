import React, { useState, useEffect } from 'react'
import axios from 'axios'

export default function App() {

  const [data, setData] = useState({gen_question: '', gen_answer: '', context:'', provided_answer: ''})
  const [showAnswer, setShowAnswer] = useState(false)
  const [loading, setLoading] = useState(true)
  const [score, setScore] = useState(-1)

  useEffect(() => {
    axios.get('http://localhost:5000/predict').then((res) => {
      console.log(res.data)
      let data = res.data
      setData({...data, gen_question: data.gen_question, gen_answer: data.gen_answer, context: data.context})
    }).catch((err) => {
      console.log(err)
    })
  }, [])

  useEffect(()=>{
    if(data.context!=='' && data.gen_answer!=='' && data.gen_question!==''){
      setLoading(false)
    }
  }, [data])

  const handleNext = ()=>{
    setLoading(true)
    setScore(-1)
    setData({gen_question: '', gen_answer: '', context:'', provided_answer: ''})
    setShowAnswer(false)
    axios.get('http://localhost:5000/predict').then((res) => {
      console.log(res.data)
      let data = res.data
      setData({...data, gen_question: data.gen_question, gen_answer: data.gen_answer, context: data.context})
    }).catch((err) => {
      console.log(err)
    })
  }

  const handleShow = ()=>{
    setShowAnswer(true)
  }

  const handleChange = (e)=>{
    setData({...data, [e.target.name]: e.target.value})
  }

  const handleSubmit = ()=>{
    axios.post('http://localhost:5000/score', {gen_answer:data.gen_answer, provided_answer:data.provided_answer}).then((res) => {
      console.log(res.data)
      let data = res.data
      setScore(data.score)
    }).catch((err) => {
      console.log(err)
    })
  }

  if(loading){
    return (
      <div className='bg-gradient-to-r from-slate-900 to-slate-700 h-screen w-full'>
        <h1 className='text-white text-3xl font-medium font-[Poppins] pt-8 mx-auto text-center pb-3 border-b-[1px] border-gray-500 w-[80%]'>CoachBOT</h1>
        <div className='flex flex-col items-center justify-center h-full w-full'>
          <h1 className='text-white text-3xl font-medium font-[Poppins]'>Loading....</h1>
        </div>
      </div>
    )
  }

  return (
    <div className='bg-gradient-to-r from-slate-900 to-slate-700 h-screen w-full'>
      <h1 className='text-white text-3xl font-medium font-[Poppins] pt-8 mx-auto text-center pb-3 border-b-[1px] border-gray-500 w-[80%]'>CoachBOT</h1>
      <div className='flex flex-col items-center justify-start h-full w-full'>
        <h1 className='text-white text-xl font-medium font-[Poppins] pt-8 mx-auto w-[80%]'>Context: <span className='text-yellow-500 text-lg'>{data.context}</span></h1>
        <h1 className='text-white text-xl font-medium font-[Poppins] pt-8 mx-auto w-[80%]'>Question: <span className='text-yellow-500 text-lg'>{data.gen_question}</span></h1>
        <input onChange={handleChange} type='text' name='provided_answer' value={data.provided_answer} className='text-black placeholder:text-gray-400 font-medium font-[Poppins] rounded-md w-[80%] py-4 mt-24 px-4' placeholder='Type your answer...'></input>
        <div className='flex flex-row items-center justify-between w-[80%] h-max mt-24'>
          <h1 onClick={handleShow} className='text-white text-xl font-medium font-[Poppins] bg-[#313131] py-3 px-4 rounded-md cursor-pointer hover:bg-[#212121]'>Show Answer</h1>
          <h1 onClick={handleSubmit} className='text-white text-xl font-medium font-[Poppins] bg-[#313131] py-3 px-4 rounded-md cursor-pointer hover:bg-[#212121]'>Submit</h1>
          <h1 onClick={handleNext} className='text-white text-xl font-medium font-[Poppins] bg-[#313131] py-3 px-4 rounded-md cursor-pointer hover:bg-[#212121]'>Next</h1>
        </div>
        {showAnswer && 
          <h1 className='text-white text-xl font-medium font-[Poppins] pt-8 mt-12 mx-auto w-[80%]'>Answer: <span className='text-yellow-500 text-lg'>{data.gen_answer}</span></h1>
        }
        {
          score!==-1 && <h1 className='text-white text-xl font-medium font-[Poppins] pt-8 mt-12 mx-auto w-[80%]'>Your Score: <span className='text-yellow-500 text-lg'>{score}</span></h1>
        }
      </div>

    </div>
  )
}
