import React from 'react'
import { 
  Card, 
  CardContent, 
  Typography, 
  Box,
  Avatar,
  LinearProgress,
  Chip,
  IconButton
} from '@mui/material'
import { 
  SmartToy, 
  PlayArrow, 
  Pause, 
  Settings,
  TrendingUp,
  TrendingDown,
  Remove
} from '@mui/icons-material'
import { motion } from 'framer-motion'

const AgentOverview = () => {
  const agents = [
    {
      id: 'agent_1',
      name: 'Research Assistant',
      status: 'active',
      task: 'Analyzing market trends',
      progress: 75,
      efficiency: 94.2,
      trend: 'up',
      capabilities: ['Web Browsing', 'AI Analysis', 'Research']
    },
    {
      id: 'agent_2', 
      name: 'Communication Hub',
      status: 'active',
      task: 'Monitoring emails & SMS',
      progress: 100,
      efficiency: 98.7,
      trend: 'up',
      capabilities: ['Email', 'SMS', 'Social Media']
    },
    {
      id: 'agent_3',
      name: 'Code Editor',
      status: 'busy',
      task: 'Refactoring backend APIs',
      progress: 45,
      efficiency: 87.3,
      trend: 'stable',
      capabilities: ['Code Generation', 'Debugging', 'Testing']
    },
    {
      id: 'agent_4',
      name: 'Data Analyst',
      status: 'idle',
      task: 'Waiting for new data',
      progress: 0,
      efficiency: 91.5,
      trend: 'down',
      capabilities: ['Data Processing', 'Analytics', 'Reporting']
    }
  ]

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return '#4ecdc4'
      case 'busy': return '#ff6b6b'
      case 'idle': return '#95a5a6'
      default: return '#667eea'
    }
  }

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'up': return <TrendingUp sx={{ fontSize: 16, color: '#4ecdc4' }} />
      case 'down': return <TrendingDown sx={{ fontSize: 16, color: '#ff6b6b' }} />
      default: return <Remove sx={{ fontSize: 16, color: '#95a5a6' }} />
    }
  }

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h5" sx={{ color: 'white', mb: 3, fontWeight: 600 }}>
          🤖 Active Agents
        </Typography>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {agents.map((agent, index) => (
            <motion.div
              key={agent.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <Box sx={{ 
                p: 2,
                borderRadius: '12px',
                background: 'rgba(255, 255, 255, 0.05)',
                border: `1px solid ${getStatusColor(agent.status)}40`,
                transition: 'all 0.3s ease',
                '&:hover': {
                  background: 'rgba(255, 255, 255, 0.08)',
                  border: `1px solid ${getStatusColor(agent.status)}60`
                }
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ 
                    background: `${getStatusColor(agent.status)}20`,
                    color: getStatusColor(agent.status),
                    mr: 2
                  }}>
                    <SmartToy />
                  </Avatar>
                  
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                      {agent.name}
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                      {agent.task}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="body2" sx={{ color: 'white', fontWeight: 600 }}>
                        {agent.efficiency}%
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        {getTrendIcon(agent.trend)}
                      </Box>
                    </Box>
                    
                    <Chip 
                      label={agent.status.toUpperCase()}
                      size="small"
                      sx={{ 
                        background: `${getStatusColor(agent.status)}20`,
                        color: getStatusColor(agent.status),
                        fontWeight: 600,
                        fontSize: '0.7rem'
                      }}
                    />

                    <IconButton size="small" sx={{ color: '#b8c5d6' }}>
                      {agent.status === 'active' ? <Pause /> : <PlayArrow />}
                    </IconButton>
                    
                    <IconButton size="small" sx={{ color: '#b8c5d6' }}>
                      <Settings />
                    </IconButton>
                  </Box>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                      Task Progress
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'white' }}>
                      {agent.progress}%
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={agent.progress}
                    sx={{
                      height: 6,
                      borderRadius: 3,
                      background: 'rgba(255, 255, 255, 0.1)',
                      '& .MuiLinearProgress-bar': {
                        background: `linear-gradient(90deg, ${getStatusColor(agent.status)}80, ${getStatusColor(agent.status)})`
                      }
                    }}
                  />
                </Box>

                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {agent.capabilities.map((capability) => (
                    <Chip
                      key={capability}
                      label={capability}
                      size="small"
                      sx={{
                        background: 'rgba(102, 126, 234, 0.2)',
                        color: '#667eea',
                        fontSize: '0.7rem'
                      }}
                    />
                  ))}
                </Box>
              </Box>
            </motion.div>
          ))}
        </Box>
      </CardContent>
    </Card>
  )
}

export default AgentOverview