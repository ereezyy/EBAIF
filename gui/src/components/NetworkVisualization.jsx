import React, { useRef, useEffect } from 'react'
import { Card, CardContent, Typography, Box } from '@mui/material'
import * as d3 from 'd3'

const NetworkVisualization = () => {
  const svgRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll("*").remove()

    const width = 800
    const height = 400

    // Sample data representing the agent network
    const nodes = [
      { id: 'central', name: 'Central AI', type: 'central', x: width/2, y: height/2 },
      { id: 'research', name: 'Research Assistant', type: 'agent', x: 200, y: 150 },
      { id: 'comm', name: 'Communication Hub', type: 'agent', x: 600, y: 150 },
      { id: 'code', name: 'Code Editor', type: 'agent', x: 200, y: 300 },
      { id: 'data', name: 'Data Analyst', type: 'agent', x: 600, y: 300 },
      { id: 'web', name: 'Web Browser', type: 'service', x: 100, y: 100 },
      { id: 'api', name: 'AI APIs', type: 'service', x: 700, y: 100 },
      { id: 'db', name: 'Database', type: 'service', x: 100, y: 350 },
      { id: 'cloud', name: 'Cloud Services', type: 'service', x: 700, y: 350 }
    ]

    const links = [
      { source: 'central', target: 'research' },
      { source: 'central', target: 'comm' },
      { source: 'central', target: 'code' },
      { source: 'central', target: 'data' },
      { source: 'research', target: 'web' },
      { source: 'comm', target: 'api' },
      { source: 'code', target: 'db' },
      { source: 'data', target: 'cloud' },
      { source: 'research', target: 'comm' },
      { source: 'code', target: 'data' }
    ]

    // Function to determine node color based on type
    const getNodeColor = (type) => {
      switch (type) {
        case 'central': return '#667eea'
        case 'agent': return '#4ecdc4'
        case 'service': return '#764ba2'
        default: return '#95a5a6'
      }
    }

    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))

    // Create container for links
    const link = svg.append('g')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', 'rgba(255, 255, 255, 0.3)')
      .attr('stroke-width', 2)
      .style('opacity', 0.6)

    // Create container for nodes
    const node = svg.append('g')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .style('cursor', 'pointer')

    // Add circles for nodes
    node.append('circle')
      .attr('r', d => d.type === 'central' ? 25 : d.type === 'agent' ? 20 : 15)
      .attr('fill', d => getNodeColor(d.type))
      .attr('stroke', 'rgba(255, 255, 255, 0.5)')
      .attr('stroke-width', 2)
      .style('filter', 'drop-shadow(0px 4px 8px rgba(0, 0, 0, 0.3))')

    // Add labels to nodes
    node.append('text')
      .text(d => d.name)
      .attr('dx', 0)
      .attr('dy', d => d.type === 'central' ? 35 : d.type === 'agent' ? 30 : 25)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .attr('font-size', '12px')
      .attr('font-weight', '500')

    // Add glow effect for important nodes
    const defs = svg.append('defs')
    const filter = defs.append('filter')
      .attr('id', 'glow')
    
    filter.append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'coloredBlur')
    
    const feMerge = filter.append('feMerge')
    feMerge.append('feMergeNode').attr('in', 'coloredBlur')
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic')

    // Apply glow to central node
    node.filter(d => d.type === 'central')
      .select('circle')
      .style('filter', 'url(#glow)')

    // Add hover effects to nodes
    node
      .on('mouseover', function(event, d) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', d => (d.type === 'central' ? 25 : d.type === 'agent' ? 20 : 15) * 1.2)
          .attr('stroke-width', 3)
      })
      .on('mouseout', function(event, d) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', d => d.type === 'central' ? 25 : d.type === 'agent' ? 20 : 15)
          .attr('stroke-width', 2)
      })

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y)

      node
        .attr('transform', d => `translate(${d.x},${d.y})`)
    })

    // Add drag behavior
    const drag = d3.drag()
      .on('start', function(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart()
        d.fx = d.x
        d.fy = d.y
      })
      .on('drag', function(event, d) {
        d.fx = event.x
        d.fy = event.y
      })
      .on('end', function(event, d) {
        if (!event.active) simulation.alphaTarget(0)
        d.fx = null
        d.fy = null
      })

    node.call(drag)

    // Clean up on unmount
    return () => {
      simulation.stop()
    }
  }, [])

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" sx={{ color: 'white', mb: 3, fontWeight: 600 }}>
          🌐 Agent Network Topology
        </Typography>
        
        <Box sx={{ 
          width: '100%', 
          height: '400px', 
          background: 'rgba(0, 0, 0, 0.2)',
          borderRadius: '12px',
          overflow: 'hidden'
        }}>
          <svg
            ref={svgRef}
            width="100%"
            height="100%"
            viewBox="0 0 800 400"
            preserveAspectRatio="xMidYMid meet"
            style={{ background: 'transparent' }}
          />
        </Box>

        <Box sx={{ mt: 2, display: 'flex', gap: 3, justifyContent: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ 
              width: 12, 
              height: 12, 
              borderRadius: '50%', 
              background: '#667eea',
              mr: 1 
            }} />
            <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
              Central AI
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ 
              width: 12, 
              height: 12, 
              borderRadius: '50%', 
              background: '#4ecdc4',
              mr: 1 
            }} />
            <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
              Agents
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ 
              width: 12, 
              height: 12, 
              borderRadius: '50%', 
              background: '#764ba2',
              mr: 1 
            }} />
            <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
              Services
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  )
}

export default NetworkVisualization