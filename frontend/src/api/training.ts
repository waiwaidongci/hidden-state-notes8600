import {request,post} from './client';export const trainingApi={list:()=>request('/trainings'),detail:(id:number)=>request('/trainings/'+id),create:(data:any)=>post('/trainings',data)};
