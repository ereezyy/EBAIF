import React, { useState } from 'react'
import { 
  Box, 
  Typography, 
  Card, 
  CardContent,
  Tabs,
  Tab,
  Chip,
  Button,
  Grid,
  LinearProgress,
  Avatar,
  IconButton
} from '@mui/material'
import { 
  Assignment, 
  Schedule, 
  CheckCircle, 
  Error,
  PlayArrow,
  Pause,
  Refresh,
  Add
} from '@mui/icons-material'
import { motion } from 'framer-motion'

const Tasks = () => {
  const [activeTab, setActiveTab] = useState(0)

  const tasks = {
    active: [
      {
        id: 'task_1',
        title: 'Market Research Analysis',
        description: 'Comprehensive analysis of Q4 market trends and competitor positioning',
        agent: 'Research Assistant',
        progress: 75,
        priority: 'high',
        startTime: '2 hours ago',
        estimatedCompletion: '45 minutes',
        subtasks: ['Data collection', 'Trend analysis', 'Report generation']
      },
      {
        id: 'task_2',
        title: 'Code Refactoring Project',
        description: 'Refactoring legacy authentication system to modern standards',
        agent: 'Code Editor',
        progress: 45,
        priority: 'medium',
        startTime: '1 hour ago',
        estimatedCompletion: '2 hours',
        subtasks: ['Dependency analysis', 'Module restructuring', 'Testing']
      },
      {
        id: 'task_3',
        title: 'Customer Email Responses',
        description: 'Processing and responding to customer inquiries',
        agent: 'Communication Hub',
        progress: 90,
        priority: 'high',
        startTime: '30 minutes ago',
        estimatedCompletion: '10 minutes',
        subtasks: ['Email parsing', 'Response generation', 'Quality check']
      }
    ],
    queued: [
      {
        id: 'task_4',
        title: 'Database Optimization',
        description: 'Optimize database queries and indexing for better performance',
        agent: 'Data Analyst',
        priority: 'medium',
        estimatedDuration: '3 hours',
        queuePosition: 1
      },
      {
        id: 'task_5',
        title: 'Security Audit',
        description: 'Comprehensive security audit of all API endpoints',
        agent: 'Code Editor',
        priority: 'high',
        estimatedDuration: '4 hours',
        queuePosition: 2
      }
    ],
    completed: [
      {
        id: 'task_6',
        title: 'Daily Backup Process',
        description: 'Automated backup of all critical system data',
        agent: 'System Manager',
        completedAt: '1 hour ago',
        duration: '15 minutes',
        priority: 'medium',
        success: true
      },
      {
        id: 'task_7',
        title: 'Social Media Monitoring',
        description: 'Monitor brand mentions across social platforms',
        agent: 'Communication Hub',
        completedAt: '3 hours ago',
        duration: '45 minutes',
        priority: 'low',
        success: true
      },
      {
        id: 'task_8',
        title: 'AI Model Training',
        description: 'Training new classification model with updated dataset',
        agent: 'ML Trainer',
        completedAt: '5 hours ago',
        duration: '2 hours',
        priority: 'high',
        success: false,
        error: 'Training data validation failed'
      }
    ]
  }

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high': return '#ff6b6b'
      case 'medium': return '#ffa726'
      case 'low': return '#4ecdc4'
      default: return '#95a5a6'
    }
  }

  const renderActiveTask = (task, index) => (
    <Grid item xs={12} key={task.id}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: index * 0.1 }}
      >
        <Card sx={{ 
          border: `1px solid ${getPriorityColor(task.priority)}40`,
          '&:hover': {
            border: `1px solid ${getPriorityColor(task.priority)}60`
          }
        }}>
          <CardContent sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'start', mb: 3 }}>
              <Avatar sx={{ 
                background: `${getPriorityColor(task.priority)}20`,
                color: getPriorityColor(task.priority),
                mr: 2
              }}>
                <Assignment />
              </Avatar>
              
              <Box sx={{ flexGrow: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h6" sx={{ color: 'white', fontWeight: 600, mr: 2 }}>
                    {task.title}
                  </Typography>
                  <Chip
                    label={task.priority.toUpperCase()}
                    size="small"
                    sx={{
                      background: `${getPriorityColor(task.priority)}20`,
                      color: getPriorityColor(task.priority),
                      fontWeight: 600
                    }}
                  />
                </Box>
                <Typography variant="body2" sx={{ color: '#b8c5d6', mb: 2 }}>
                  {task.description}
                </Typography>
                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                  <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                    Agent: <span style={{ color: 'white' }}>{task.agent}</span>
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                    Started: <span style={{ color: 'white' }}>{task.startTime}</span>
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                    ETA: <span style={{ color: 'white' }}>{task.estimatedCompletion}</span>
                  </Typography>
                </Box>
              </Box>

              <Box sx={{ display: 'flex', gap: 1 }}>
                <IconButton size="small" sx={{ color: '#4ecdc4' }}>
                  <Pause />
                </IconButton>
                <IconButton size="small" sx={{ color: '#b8c5d6' }}>
                  <Refresh />
                </IconButton>
              </Box>
            </Box>

            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                  Progress
                </Typography>
                <Typography variant="body2" sx={{ color: 'white', fontWeight: 600 }}>
                  {task.progress}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={task.progress}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  background: 'rgba(255, 255, 255, 0.1)',
                  '& .MuiLinearProgress-bar': {
                    background: `linear-gradient(90deg, ${getPriorityColor(task.priority)}80, ${getPriorityColor(task.priority)})`
                  }
                }}
              />
            </Box>

            <Box>
              <Typography variant="subtitle2" sx={{ color: 'white', mb: 1, fontWeight: 600 }}>
                Subtasks
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {task.subtasks.map((subtask, idx) => (
                  <Chip
                    key={subtask}
                    label={subtask}
                    size="small"
                    icon={idx < Math.floor(task.progress / 30) ? <CheckCircle /> : <Schedule />}
                    sx={{
                      background: idx < Math.floor(task.progress / 30) 
                        ? 'rgba(76, 175, 80, 0.2)' 
                        : 'rgba(255, 255, 255, 0.1)',
                      color: idx < Math.floor(task.progress / 30) ? '#4caf50' : '#b8c5d6'
                    }}
                  />
                ))}
              </Box>
            </Box>
          </CardContent>
        </Card>
      </motion.div>
    </Grid>
  )

  const renderQueuedTask = (task, index) => (
    <Grid item xs={12} md={6} key={task.id}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: index * 0.1 }}
      >
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Avatar sx={{ 
                background: 'rgba(255, 167, 38, 0.2)',
                color: '#ffa726',
                mr: 2
              }}>
                <Schedule />
              </Avatar>
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                  {task.title}
                </Typography>
                <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                  Queue Position: #{task.queuePosition}
                </Typography>
              </Box>
              <Chip
                label={task.priority.toUpperCase()}
                size="small"
                sx={{
                  background: `${getPriorityColor(task.priority)}20`,
                  color: getPriorityColor(task.priority),
                  fontWeight: 600
                }}
              />
            </Box>
            <Typography variant="body2" sx={{ color: '#b8c5d6', mb: 2 }}>
              {task.description}
            </Typography>
            <Box sx={{ display: 'flex', justify: 'space-between' }}>
              <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                Agent: <span style={{ color: 'white' }}>{task.agent}</span>
              </Typography>
              <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                Est. Duration: <span style={{ color: 'white' }}>{task.estimatedDuration}</span>
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </motion.div>
    </Grid>
  )

  const renderCompletedTask = (task, index) => (
    <Grid item xs={12} md={6} key={task.id}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: index * 0.1 }}
      >
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Avatar sx={{ 
                background: task.success ? 'rgba(76, 175, 80, 0.2)' : 'rgba(255, 107, 107, 0.2)',
                color: task.success ? '#4caf50' : '#ff6b6b',
                mr: 2
              }}>
                {task.success ? <CheckCircle /> : <Error />}
              </Avatar>
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                  {task.title}
                </Typography>
                <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                  Completed {task.completedAt}
                </Typography>
              </Box>
              <Chip
                label={task.success ? 'SUCCESS' : 'FAILED'}
                size="small"
                sx={{
                  background: task.success ? 'rgba(76, 175, 80, 0.2)' : 'rgba(255, 107, 107, 0.2)',
                  color: task.success ? '#4caf50' : '#ff6b6b',
                  fontWeight: 600
                }}
              />
            </Box>
            <Typography variant="body2" sx={{ color: '#b8c5d6', mb: 2 }}>
              {task.description}
            </Typography>
            {!task.success && task.error && (
              <Box sx={{ 
                p: 2, 
                borderRadius: '8px', 
                background: 'rgba(255, 107, 107, 0.1)',
                border: '1px solid rgba(255, 107, 107, 0.3)',
                mb: 2
              }}>
                <Typography variant="body2" sx={{ color: '#ff6b6b' }}>
                  Error: {task.error}
                </Typography>
              </Box>
            )}
            <Box sx={{ display: 'flex', justify: 'space-between' }}>
              <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                Agent: <span style={{ color: 'white' }}>{task.agent}</span>
              </Typography>
              <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                Duration: <span style={{ color: 'white' }}>{task.duration}</span>
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </motion.div>
    </Grid>
  )

  return (
    <Box sx={{ p: 3 }}>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
          <Typography variant="h3" sx={{ color: 'white', fontWeight: 700 }}>
            📋 Task Management
          </Typography>
          <Button
            variant="contained"
            startIcon={<Add />}
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              borderRadius: '12px',
              px: 3
            }}
          >
            Create Task
          </Button>
        </Box>
      </motion.div>

      <Card sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={(e, newValue) => setActiveTab(newValue)}
          sx={{
            '& .MuiTab-root': {
              color: '#b8c5d6',
              textTransform: 'none',
              fontSize: '1rem',
              fontWeight: 500
            },
            '& .Mui-selected': {
              color: '#667eea'
            },
            '& .MuiTabs-indicator': {
              backgroundColor: '#667eea'
            }
          }}
        >
          <Tab 
            label={`Active (${tasks.active.length})`} 
            icon={<PlayArrow />}
            iconPosition="start"
          />
          <Tab 
            label={`Queued (${tasks.queued.length})`} 
            icon={<Schedule />}
            iconPosition="start"
          />
          <Tab 
            label={`Completed (${tasks.completed.length})`} 
            icon={<CheckCircle />}
            iconPosition="start"
          />
        </Tabs>
      </Card>

      <Box>
        {activeTab === 0 && (
          <Grid container spacing={3}>
            {tasks.active.map(renderActiveTask)}
          </Grid>
        )}
        {activeTab === 1 && (
          <Grid container spacing={3}>
            {tasks.queued.map(renderQueuedTask)}
          </Grid>
        )}
        {activeTab === 2 && (
          <Grid container spacing={3}>
            {tasks.completed.map(renderCompletedTask)}
          </Grid>
        )}
      </Box>
    </Box>
  )
}

export default Tasks