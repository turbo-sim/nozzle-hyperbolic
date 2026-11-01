/* Saved perfect-gas Sod plotting conversion; no solver or property call. */
  function renderSodProfiles(){
    const data=JSON.parse($('roe-sod-data').textContent);
    const ids=['roe-sod-pressure','roe-sod-density','roe-sod-velocity'];
    if(!data)return;
    beginPlots();
    const rows=array(data.records).filter(row=>row.cells===Number($('roe-sod-grid').value));
    const physical=data.settings?.physical;
    if(!rows.length||!physical){ids.forEach(id=>draw(id,[],{},'No saved Sod profile for this grid.'));finishPlots();return;}
    const first=rows[0],q=first.exact_final_conservative_cell_averages;
    if(!q?.length){ids.forEach(id=>draw(id,[],{},'Exact conservative cell averages are not retained.'));finishPlots();return;}
    const left=physical.left,right=physical.right;
    const internalLeft=q[0][2]/left.rho-.5*left.u**2;
    const internalRight=q.at(-1)[2]/right.rho-.5*right.u**2;
    const coefficient=(internalLeft-internalRight)/(left.p/left.rho-right.p/right.rho);
    const offset=internalLeft-coefficient*left.p/left.rho;
    const gamma=1+1/coefficient,sound=Math.sqrt(gamma*left.p/left.rho);
    const x=first.physical_cell_centers.map(value=>value/physical.length);
    const reference=[q.map(u=>(u[2]-.5*u[1]**2/u[0]-offset*u[0])/coefficient/left.p),q.map(u=>u[0]/left.rho),q.map(u=>u[1]/u[0]/sound)];
    const names={euler_explicit:'Explicit Euler',rk3:'SSPRK3',implicit_coupled:'Backward Euler · coupled BC',implicit_segregated:'Backward Euler · segregated BC'};
    const methods=['euler_explicit','rk3','implicit_coupled','implicit_segregated'];
    const colors=['#6a00a8','#b12a90','#e16462','#fca636'];
    const dashes=['solid','dash','dashdot','dot'];
    const labels=['p / pL','ρ / ρL','u / aL'];
    const title=['Pressure','Density','Velocity'];
    ids.forEach((id,field)=>{
      const traces=[{type:'scatter',mode:'lines',name:'Exact conservative cell averages',x,y:reference[field],line:{color:palette().ink,dash:'dot',width:2}}];
      rows.forEach(row=>{
        const index=Math.max(0,methods.indexOf(row.method?.id));
        const state=array(row.final_state).slice(1,-1);
        const values=state.map(w=>field===0?w[0]/left.p:field===1?(coefficient+1)*w[0]/(w[2]-offset)/left.rho:w[1]/sound);
        traces.push({type:'scatter',mode:'lines',name:names[row.method?.id]||row.method?.label||row.method?.id,x:row.physical_cell_centers.map(value=>value/physical.length),y:values,line:{color:colors[index],dash:dashes[index],width:2},meta:{saved_cells:row.cells,saved_method:row.method?.id,numerical_passed:row.classification?.numerical_passed,strict_criterion_passed:row.passed}});
      });
      draw(id,traces,{title:{text:title[field],font:{size:17}},xaxis:{title:'x / L',range:[0,1]},yaxis:{title:labels[field]}});
    });
    $('roe-sod-profile-caption').textContent=`${first.cells} physical cells · CFL 0.5 · t = ${(first.time*1000).toPrecision(5)} ms. Curves use the saved final states and exact conservative cell averages. The affine perfect-gas energy relation is recovered from the saved unchanged end states, including its energy reference offset; no EOS or numerical solve is evaluated.`;
    finishPlots();
  }
